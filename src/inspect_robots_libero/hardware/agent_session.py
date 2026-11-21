"""Human-operated multi-scene evaluation support; no devices open on import."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import select
import shutil
import sys
from zoneinfo import ZoneInfo

from inspect_robots.scene import Scene

from .embodiment import SO101HardwareEmbodiment


class OperatorFault(RuntimeError):
    pass


class SessionQuit(Exception):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scene_queue(catalog_path, queue_path):
    catalog_path=Path(catalog_path).resolve()
    catalog=json.loads(catalog_path.read_text())
    queue=json.loads(Path(queue_path).read_text())
    if queue.get("schema_version")!=1 or queue.get("subset") not in ("ID","OOD","AB"):
        raise ValueError("Queue needs schema_version=1 and an explicit subset")
    by_id={}
    for entry in catalog["scenes"]:
        path=(catalog_path.parent/entry["config"]).resolve()
        scene=json.loads(path.read_text())
        by_id[scene["id"]]=(path,scene)
    scenes=[]
    seen=set()
    for task in queue["tasks"]:
        for variation in task["scenes"]:
            ident=f"{queue['subset']}_task{task['task_id']:02d}_v{variation:02d}"
            if ident in seen:
                raise ValueError(f"Duplicate queue scene {ident}")
            seen.add(ident)
            if ident not in by_id:
                raise ValueError(f"Catalog is missing {ident}")
            path,scene=by_id[ident]
            reference=Path(scene["provenance"]["reference_image"])
            if not reference.is_file():
                raise ValueError(f"Missing reference photo for {ident}: {reference}")
            scenes.append(dict(scene,scene_config=str(path),scene_order=len(scenes)+1,
                source_scene_file=str(path),source_scene_sha256=digest(path),
                source_reference_file=str(reference),source_reference_sha256=digest(reference)))
    if not scenes:
        raise ValueError("Session queue is empty")
    return scenes


def create_session_directory(root, agent, effort, timezone="America/Chicago"):
    root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=True)
    name=re.sub(r"[^A-Za-z0-9._-]+","_",agent).strip("._")
    if not name:
        raise ValueError("Agent name must contain a filename-safe character")
    stamp=datetime.now(ZoneInfo(timezone)).strftime("%Y%m%d_%H%M%S%z")
    base=f"{stamp}_{name}_{effort}"
    for index in range(1000):
        path=root/(base if index==0 else f"{base}_{index:02d}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise ValueError("Cannot allocate a unique session output directory")


def freeze_inputs(output,settings,profile,scenes,initial_pose):
    """Copy local configuration and references; kochIRVL remains read-only."""
    output=Path(output)
    calibration_dir=output/"calibration"
    calibration_dir.mkdir()
    calibration=calibration_dir/(settings["follower_id"]+".json")
    shutil.copy2(settings["calibration_file"],calibration)
    reset=output/"initial_pose.json"
    shutil.copy2(initial_pose,reset)
    profile_path=output/"hardware_profile.json"
    urdf=output/"robot.urdf"
    shutil.copy2(profile["urdf"],urdf)
    profile_path.write_text(json.dumps(dict(profile,urdf=str(urdf)),indent=2))
    base=dict(settings,profile_file=str(profile_path),calibration_file=str(calibration),
              reset_mode="none",reset_action_file=str(reset))
    paths={}
    for kind,overlay in (("episode",False),("operator",True)):
        path=output/f"hardware_{kind}.json"
        path.write_text(json.dumps(dict(base,reference_overlay=overlay),indent=2))
        paths[kind]=path
    frozen=[]
    for scene in scenes:
        directory=output/"scenes"/scene["id"]
        directory.mkdir(parents=True)
        reference=directory/("reference"+Path(scene["source_reference_file"]).suffix)
        shutil.copy2(scene["source_reference_file"],reference)
        data=json.loads(Path(scene["source_scene_file"]).read_text())
        data["provenance"]["reference_image"]=str(reference)
        path=directory/"scene.json"
        path.write_text(json.dumps(data,indent=2))
        frozen.append(dict(scene,scene_config=str(path),reference_image=str(reference)))
    return paths,frozen


def new_attempt(scene,attempt,model,effort,accounting_mode,hardware_mode,hz,calibration_status):
    child_id=f"{scene['id']}_r01_m00_{effort}"
    directory=f"task{scene['task_id']:02d}_scene{scene['variation']:02d}/attempt{attempt:02d}"
    return dict(id=f"{scene['id']}_a{attempt:02d}_{effort}",run_folder=f"{directory}/{child_id}",
        episode_directory=directory,child_run_id=child_id,benchmark="hardware",subset=scene["subset"],
        task_id=scene["task_id"],variation=scene["variation"],repeat=1,attempt=attempt,scene_order=scene["scene_order"],
        instruction=scene["instruction"],scene_id=scene["id"],model=model,effort=effort,
        accounting_mode=accounting_mode,device_mode=hardware_mode,control_interface="cartesian",control_hz=hz,
        fidelity={"mock":"mock_hardware","dry-run":"physical_observation_dry_run","execute":"physical_robot"}[hardware_mode],
        calibration_status=calibration_status,status="pending",steps=None,environment_success=None,human_success=None,
        success_source="unscored")


def ingest_attempt(output,row):
    path=Path(output)/row["episode_directory"]/"evaluation.json"
    if not path.exists():
        row.update(status="error",error_type="EpisodeReportMissing",usage_unknown=True)
        return
    child=json.loads(path.read_text())
    if len(child["runs"])!=1:
        raise ValueError("Session child evaluation must contain exactly one attempt")
    saved={key:row[key] for key in ("id","run_folder","episode_directory","child_run_id","attempt","scene_order")}
    row.update(child["runs"][0])
    row.update(saved)


class Terminal:
    def __init__(self,script=None):
        self.script=iter(script) if script is not None else None

    def read(self,prompt,monitor=None):
        if self.script is not None:
            try:
                answer=next(self.script)
            except StopIteration:
                raise SessionQuit("Operator script ended") from None
            print(prompt+(answer or "<ENTER>"),flush=True)
            return answer.strip().lower()
        if monitor is None:
            try:
                return input(prompt).strip().lower()
            except EOFError:
                raise SessionQuit("Terminal input closed") from None
        print(prompt,end="",flush=True)
        while True:
            monitor()
            if select.select([sys.stdin],[],[],.1)[0]:
                answer=sys.stdin.readline()
                if not answer:
                    raise SessionQuit("Terminal input closed")
                return answer.strip().lower()


class OperatorControl:
    def __init__(self,output,config,mode,assume_calibration,video):
        self.output=Path(output)
        self.config=config
        self.mode=mode
        self.assume_calibration=assume_calibration
        self.video=video
        self.env=None
        self.last_env=None
        self.stage=0
        self.reset_ready=False

    def view(self,scene,phase):
        if self.env is None:
            self.stage+=1
            directory=self.output/"operator"/f"{self.stage:03d}_{scene['id']}_{phase}"
            env=SO101HardwareEmbodiment(config_path=self.config,mode=self.mode,assume_calibration=self.assume_calibration,
                output_dir=directory,max_steps=1800,policy_seconds=180,auto_start=True,video=self.video)
            self.env=self.last_env=env
            try:
                env.reset(Scene(id=scene["id"],instruction=scene["instruction"],
                    metadata={"replica_scene_config":scene["scene_config"]}))
            except BaseException:
                self.close()
                raise
        try:
            reply=self.env._rpc(dict(op="operator_view",reference_image=scene["reference_image"],label=f"{phase}: {scene['id']}"))
            self.env._observation(reply)
        except Exception as exc:
            raise OperatorFault(str(exc)) from exc

    def monitor(self):
        if self.env is None or self.env.process is None or self.env.process.poll() is not None:
            raise OperatorFault("Operator camera/control worker stopped; no new trial will start")

    def reset(self):
        self.monitor()
        self.reset_ready=False
        try:
            reply=self.env._rpc(dict(op="operator_reset"))
            self.env._observation(reply)
        except Exception as exc:
            raise OperatorFault(str(exc)) from exc
        if reply.get("stopped") or not reply["feedback"]["accepted"]:
            raise OperatorFault(reply.get("stopped") or reply["feedback"]["detail"])
        self.reset_ready=True
        return reply

    def close(self):
        if self.env is not None:
            env,self.env=self.env,None
            try:
                env.close()
            finally:
                (env.output/"trajectory.json").write_text(json.dumps(env.trajectory))

    def before_trial(self):
        self.close()
        self.reset_ready=False
