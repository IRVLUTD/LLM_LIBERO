"""Terminal-operated SO-101 session over the selected 15 Replica task scenes.

Default: create a timestamped plan without API/device access. Explicit --mock
or --execute-paid-rollouts starts the interactive session. r+Enter resets,
Enter starts, s/f confirms the result, and n/t/k advances/retries/skips.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import run_replica_evaluation as single
from inspect_robots._dotenv import read_dotenv
from inspect_robots_libero.evaluation_report import write_reports
from inspect_robots_libero.hardware.agent_session import (
    OperatorControl, OperatorFault, SessionQuit, Terminal, create_session_directory,
    digest, freeze_inputs, ingest_attempt, load_scene_queue, new_attempt,
)
from inspect_robots_libero.hardware.kinematics import MOTORS, calibrated_motor_limits, load_profile
from inspect_robots_libero.hardware.settings import load_settings, execution_gate

ROOT=Path(__file__).resolve().parents[1]


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--catalog",type=Path,default=ROOT/"outputs/replica-scenes/catalog.json")
    p.add_argument("--queue",type=Path,default=ROOT/"configs/vla_replica_agent_session.json")
    p.add_argument("--hardware-config",type=Path,required=True)
    p.add_argument("--initial-pose",type=Path,help="Defaults to hardware config's reset_action_file")
    p.add_argument("--hardware-mode",choices=("mock","dry-run","execute"),default="mock")
    p.add_argument("--model",default="openai/gpt-6.1-sol")
    p.add_argument("--agent",help="Optional folder label; defaults to provider/model")
    p.add_argument("--effort",choices=single.EFFORTS,default="medium")
    p.add_argument("--max-calls",type=int,default=80)
    p.add_argument("--max-steps",type=int,default=1800)
    p.add_argument("--max-output-tokens",type=int,default=8192)
    p.add_argument("--policy-seconds",type=float,default=900)
    p.add_argument("--per-run-budget",type=float,help="USD cap for each attempt, including retries")
    p.add_argument("--total-budget",type=float,help="USD cap for the entire session, including retries")
    p.add_argument("--pricing-file",type=Path)
    p.add_argument("--assume-calibration",action="store_true")
    p.add_argument("--no-video",action="store_true")
    p.add_argument("--keep-torque-at-end",action="store_true",help="Skip the session-end release prompt")
    p.add_argument("--output-root",type=Path,default=ROOT/"outputs/VLAReplica_Agent")
    p.add_argument("--timezone",default="America/Chicago",help="Timezone used in output folder timestamps")
    p.add_argument("--operator-script",type=Path,help="Mock-only JSON array of terminal answers for offline checks")
    modes=p.add_mutually_exclusive_group()
    modes.add_argument("--mock",action="store_true")
    modes.add_argument("--execute-paid-rollouts",action="store_true")
    return p


class AgentSession:
    def __init__(self,args,output,scenes,configs,data,terminal,run_episode=single.main):
        self.args,self.output,self.scenes,self.configs=args,Path(output),scenes,configs
        self.data,self.terminal,self.run_episode=data,terminal,run_episode
        self.control=OperatorControl(output,configs["operator"],args.hardware_mode,args.assume_calibration,not args.no_video)
        self.current_row=None

    def event(self,kind,**values):
        with (self.output/"session_events.jsonl").open("a") as file:
            payload=dict(values)
            payload.update(event=kind,created_utc=payload.get("created_utc",datetime.now(timezone.utc).isoformat()))
            file.write(json.dumps(payload)+"\n")

    def save(self):
        self.data["budget_charge_usd"]=sum(r.get("budget_charge_usd") or 0 for r in self.data["runs"])
        path=self.output/"evaluation.json"
        temp=path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self.data,indent=2))
        temp.replace(path)
        write_reports(self.output)

    def remaining(self):
        return self.data["protocol"]["total_budget_usd"]-self.data["budget_charge_usd"]

    def view(self,scene,phase):
        while True:
            try:
                self.control.view(scene,phase)
                return
            except Exception as exc:
                self.control.close()
                self.control.reset_ready=False
                self.event("operator_fault",scene_id=scene["id"],detail=str(exc))
                print(f"[PAUSED] Camera/control connection failed: {exc}",flush=True)
                answer=self.terminal.read("c=retry connection, q=finish session: ")
                if answer!="c":
                    raise SessionQuit("Operator ended session after a control fault")

    def command(self,prompt,scene,phase):
        try:
            return self.terminal.read(prompt,monitor=self.control.monitor)
        except OperatorFault as exc:
            self.control.close()
            self.control.reset_ready=False
            self.event("operator_fault",scene_id=scene["id"],detail=str(exc))
            print(f"[PAUSED] {exc}",flush=True)
            answer=self.terminal.read("c=retry connection, q=finish session: ")
            if answer!="c":
                raise SessionQuit("Operator ended session after a control fault")
            self.view(scene,phase)
            return "__reconnected__"

    def reset(self,scene,phase):
        print("[RESET] Moving to the saved initial pose.",flush=True)
        self.event("reset_requested",scene_id=scene["id"],phase=phase)
        try:
            reply=self.control.reset()
            self.event("reset_finished",scene_id=scene["id"],phase=phase,reset_steps=reply["reset_control_steps"],mode=self.args.hardware_mode)
            print("[RESET] Initial-pose command completed." if self.args.hardware_mode!="dry-run" else "[RESET] Dry-run reset planned; no motors moved.",flush=True)
        except Exception as exc:
            self.control.reset_ready=False
            self.control.close()
            self.event("reset_fault",scene_id=scene["id"],detail=str(exc))
            print(f"[PAUSED] Reset failed: {exc}",flush=True)
            self.view(scene,phase)

    def setup(self,scene):
        self.view(scene,"SETUP")
        print(f"\n[{scene['scene_order']}/{len(self.scenes)}] {scene['id']}: {scene['instruction']}",flush=True)
        print(f"Reference overlay: {scene['reference_image']}",flush=True)
        print("Before r+Enter: clear the reset path and remove/support held objects.",flush=True)
        while True:
            answer=self.command("SETUP: r=reset arm, ENTER=start after arranging scene, k=skip, q=finish: ",scene,"SETUP")
            if answer=="r":
                self.reset(scene,"SETUP")
            elif not answer:
                if not self.control.reset_ready:
                    print("Press r+Enter first to establish the initial arm pose.",flush=True)
                    continue
                return True
            elif answer=="k":
                return False
            elif answer=="q":
                raise SessionQuit("Operator ended session during setup")

    def run_attempt(self,scene,row):
        self.control.before_trial()
        self.current_row=row
        directory=self.output/row["episode_directory"]
        cap=min(self.args.per_run_budget,self.remaining())
        argv=["--benchmark","hardware","--scene-json",scene["scene_config"],
            "--hardware-config",str(self.configs["episode"]),"--hardware-mode",self.args.hardware_mode,
            "--model",self.args.model,"--effort",self.args.effort,"--max-calls",str(self.args.max_calls),
            "--max-steps",str(self.args.max_steps),"--max-output-tokens",str(self.args.max_output_tokens),
            "--policy-seconds",str(self.args.policy_seconds),"--per-run-budget",str(cap),
            "--total-budget",str(self.remaining()),"--output",str(directory),"--auto-start","--keep-torque"]
        argv.append("--mock" if self.args.mock else "--execute-paid-rollouts")
        if self.args.assume_calibration:
            argv.append("--assume-calibration")
        if self.args.no_video:
            argv.append("--no-video")
        if self.args.pricing_file:
            argv.extend(["--pricing-file",str(self.args.pricing_file.resolve())])
        row["status"]="running"
        self.event("attempt_started",run_id=row["id"],scene_id=scene["id"],per_attempt_budget=cap)
        self.save()
        try:
            self.run_episode(argv)
        except SystemExit as exc:
            self.event("attempt_process_error",run_id=row["id"],exit_code=exc.code)
        except Exception as exc:
            self.event("attempt_exception",run_id=row["id"],error_type=type(exc).__name__,detail=str(exc))
        finally:
            ingest_attempt(self.output,row)
            self.event("attempt_finished",run_id=row["id"],status=row["status"],budget_charge_usd=row.get("budget_charge_usd"))
            self.save()
            self.current_row=None

    def score(self,scene,row):
        self.view(scene,"RESULT")
        print(f"[RESULT] Model termination: {row.get('termination_reasons')}; status: {row['status']}",flush=True)
        if row.get("hardware_stop_detail") or row["status"] in ("error","hardware_stopped","interrupted"):
            self.event("hardware_fault_pause",run_id=row["id"],detail=row.get("hardware_stop_detail") or row.get("error_type") or row["status"])
            print(f"[HARDWARE/ERROR PAUSE] {row.get('hardware_stop_detail') or row.get('error_type') or row['status']}",flush=True)
        while True:
            answer=self.command("RESULT: s=confirmed success, f=failure, u=unscored, q=finish unscored: ",scene,"RESULT")
            if answer=="q":
                raise SessionQuit("Operator ended session before scoring")
            if answer in ("s","f","u"):
                break
        verdict={"s":"success","f":"failure","u":"unscored"}[answer]
        physical=self.args.hardware_mode=="execute"
        label=(answer=="s") if physical and answer!="u" else None
        row.update(operator_verdict=verdict,human_success=label,
                   success_source="human" if label is not None else "unscored" if physical else "mock_or_dryrun_operator")
        audit=dict(created_utc=datetime.now(timezone.utc).isoformat(),run_id=row["id"],child_run_id=row["child_run_id"],
                   operator_verdict=verdict,human_success=label)
        child_dir=self.output/row["episode_directory"]
        child_dir.mkdir(parents=True,exist_ok=True)
        for path in (self.output/"human_labels.jsonl",child_dir/"human_labels.jsonl"):
            with path.open("a") as file:
                file.write(json.dumps(audit)+"\n")
        child_path=child_dir/"evaluation.json"
        if child_path.exists():
            child=json.loads(child_path.read_text())
            child["runs"][0].update({k:row[k] for k in ("operator_verdict","human_success","success_source")})
            child_path.write_text(json.dumps(child,indent=2))
            write_reports(child_dir)
        self.event("human_verdict",**audit)
        self.save()
        print("Before r+Enter: clear the reset path and remove/support held objects.",flush=True)
        return verdict

    def after_attempt(self,scene,row,verdict):
        uncertain=row.get("usage_unknown",False)
        while True:
            if uncertain:
                prompt="BILLING UNKNOWN: r=reset, q/ENTER=finish; no further API attempts: "
            elif verdict=="success":
                prompt="SUCCESS: r=reset initial pose, n/ENTER=next (after reset), q=finish: "
            else:
                prompt="REVIEW: r=reset initial pose, t=retry (after reset), k=skip scene, q=finish: "
            answer=self.command(prompt,scene,"RESULT")
            if answer=="r":
                self.reset(scene,"RESULT")
            elif uncertain and answer in ("","q"):
                self.data["status"]="stopped_usage_unknown"
                raise SessionQuit("Usage unknown; further API calls blocked")
            elif answer=="q":
                raise SessionQuit("Operator ended session after scoring")
            elif not uncertain and ((verdict=="success" and answer in ("","n")) or answer=="t"):
                if not self.control.reset_ready:
                    print("Reset with r+Enter before the next attempt.",flush=True)
                    continue
                return "retry" if answer=="t" else "next"
            elif not uncertain and answer=="k":
                return "next"

    def run(self):
        self.data["status"]="running"
        self.save()
        try:
            for scene in self.scenes:
                row=next(r for r in self.data["runs"] if r["scene_id"]==scene["id"] and r["attempt"]==1)
                while True:
                    if self.remaining()<=0:
                        self.data["status"]="budget_exhausted"
                        raise SessionQuit("Session API budget exhausted")
                    if not self.setup(scene):
                        row.update(status="not_run",operator_verdict="skipped_before_start")
                        self.event("scene_skipped",scene_id=scene["id"])
                        self.save()
                        break
                    self.run_attempt(scene,row)
                    verdict=self.score(scene,row)
                    action=self.after_attempt(scene,row,verdict)
                    if action=="next":
                        break
                    new=new_attempt(scene,row["attempt"]+1,self.args.model,self.args.effort,
                        row["accounting_mode"],self.args.hardware_mode,row["control_hz"],row["calibration_status"])
                    self.data["runs"].insert(self.data["runs"].index(row)+1,new)
                    row=new
                    self.save()
            else:
                self.data["status"]="completed"
        except SessionQuit as exc:
            if self.data["status"]=="running":
                self.data["status"]="stopped_operator"
            self.event("session_stopped",reason=str(exc))
        except KeyboardInterrupt:
            self.data["status"]="interrupted"
            self.event("session_interrupted")
        finally:
            try:
                self.control.close()
            finally:
                for row in self.data["runs"]:
                    if row["status"]=="pending":
                        row["status"]="not_run"
                if self.data["status"]=="running":
                    self.data["status"]="error"
                self.save()
                shutdown={}
                try:
                    single.prompt_hardware_torque_release(self.control.last_env,shutdown,keep_torque=self.args.keep_torque_at_end)
                finally:
                    self.data["session_shutdown"]=shutdown
                    self.data["protocol"]["session_shutdown"]=shutdown
                    self.event("session_shutdown",**shutdown)
                    self.save()


def main(argv=None):
    p=parser()
    args=p.parse_args(argv)
    try:
        for key in ("max_calls","max_steps","max_output_tokens"):
            if getattr(args,key)<1:
                raise ValueError(f"{key} must be positive")
        if args.max_output_tokens>128000 or not math.isfinite(args.policy_seconds) or args.policy_seconds<=0:
            raise ValueError("Invalid output token or episode time limit")
        if args.mock and args.hardware_mode=="execute":
            raise ValueError("Mock LLM responses cannot drive physical motors")
        if args.operator_script and not (args.mock and args.hardware_mode=="mock"):
            raise ValueError("--operator-script is restricted to mock LLMs and mock hardware")
        if args.execute_paid_rollouts and (args.total_budget is None or args.per_run_budget is None):
            raise ValueError("Paid sessions require --total-budget and --per-run-budget")
        args.total_budget=args.total_budget if args.total_budget is not None else 100.
        args.per_run_budget=args.per_run_budget if args.per_run_budget is not None else 5.
        if any(not math.isfinite(v) or v<=0 for v in (args.total_budget,args.per_run_budget)):
            raise ValueError("Budgets must be positive finite USD values")
        if args.pricing_file:
            pricing=json.loads(args.pricing_file.read_text())
            for name,rates in pricing.items():
                if name.partition("/")[0] not in single.KEYS or not isinstance(rates,list) or len(rates)!=4 or any(
                        not isinstance(v,(float,int)) or isinstance(v,bool) or not math.isfinite(v) or v<0 for v in rates):
                    raise ValueError(f"Invalid price row {name}")
                if rates[2]<max(rates[:2]):
                    raise ValueError("Cache-write reservation must cover input rates")
                single.PRICING[name]=tuple(rates)
        if args.model not in single.PRICING:
            raise ValueError("Unknown model pricing; supply --pricing-file")
        settings,raw_profile=load_settings(args.hardware_config)
        execution_gate(settings,raw_profile,args.hardware_mode,args.assume_calibration)
        profile=load_profile(settings["profile_file"])
        initial=args.initial_pose or settings.get("reset_action_file")
        if not initial:
            raise ValueError("Provide --initial-pose or a configured reset_action_file")
        initial=Path(initial).resolve()
        reset=json.loads(initial.read_text())
        if set(reset)!=set(MOTORS) or any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) for v in reset.values()):
            raise ValueError("Initial pose needs exactly six finite .pos values")
        limits=calibrated_motor_limits(json.loads(Path(settings["calibration_file"]).read_text()))
        if any(not limits[i,0]<=reset[name]<=limits[i,1] for i,name in enumerate(MOTORS)):
            raise ValueError("Initial pose exceeds calibrated motor travel")
        scenes=load_scene_queue(args.catalog,args.queue)
        execute=args.mock or args.execute_paid_rollouts
        if execute and args.hardware_mode!="mock" and not single.sys.stdin.isatty():
            raise ValueError("Real-device sessions require an interactive terminal")
        if execute and args.hardware_mode!="mock" and not os.environ.get("DISPLAY"):
            raise ValueError("Scene setup overlays require desktop DISPLAY")
        if args.execute_paid_rollouts:
            env={**read_dotenv(ROOT/".env"),**os.environ}
            if not env.get(single.KEYS[args.model.partition("/")[0]]):
                raise ValueError("Missing model provider API key")
        script=json.loads(args.operator_script.read_text()) if args.operator_script else None
        if script is not None and (not isinstance(script,list) or not all(isinstance(s,str) for s in script)):
            raise ValueError("Operator script must be a JSON array of terminal answers")
        os.umask(0o077)
        output=create_session_directory(args.output_root,args.agent or args.model,args.effort,args.timezone)
    except (OSError,ValueError,KeyError) as exc:
        p.error(str(exc))
    settings.update(video=not args.no_video,preview=args.hardware_mode!="mock")
    if args.hardware_mode=="mock":
        settings["mock_state_file"]=str(output/"mock_motor_state.json")
    configs,scenes=freeze_inputs(output,settings,profile,scenes,initial)
    mode="mock" if args.mock else "paid" if args.execute_paid_rollouts else "planned"
    calibration_status="assumed" if args.assume_calibration else "verified_profile" if profile.get("kinematics_verified") and profile.get("world_frame_verified") else "unverified"
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),benchmark="hardware",runner="VLAReplica_Agent",
        accounting_mode=mode,model=args.model,models=[args.model],effort=args.effort,efforts=[args.effort],
        agent=args.agent or args.model,folder_timezone=args.timezone,scene_queue=scenes,
        hardware_mode=args.hardware_mode,hardware_config=settings,hardware_profile=profile,
        hardware_config_sha256=digest(args.hardware_config),initial_pose_sha256=digest(initial),
        queue_sha256=digest(args.queue),catalog_sha256=digest(args.catalog),
        control_hz=profile["control_hz"],total_budget_usd=args.total_budget,per_run_budget_usd=args.per_run_budget,
        max_calls=args.max_calls,max_steps=args.max_steps,max_output_tokens=args.max_output_tokens,policy_seconds=args.policy_seconds,
        image_horizon=2,zero_shot=True,pricing_per_million={args.model:single.PRICING[args.model]},
        pricing_source=str(args.pricing_file.resolve()) if args.pricing_file else "Inherited repository price snapshot; provider billing is authoritative",
        script_sha256=digest(__file__),session_module_sha256=digest(ROOT/"src/inspect_robots_libero/hardware/agent_session.py"),
        success_scoring="Human terminal verdicts; model done is unverified; mock/dry-run labels are not physical success",
        setup="Reference overlay and manual reset; setup/reset videos and counters separate from agent attempts",
        torque="Hold between scenes; release prompt once at session end")
    data=dict(protocol=protocol,status="planned",budget_charge_usd=0.,runs=[new_attempt(scene,1,args.model,args.effort,
        mode,args.hardware_mode,profile["control_hz"],calibration_status) for scene in scenes])
    session=AgentSession(args,output,scenes,configs,data,Terminal(script),run_episode=single.main)
    session.save()
    print(f"Session folder: {output}\nQueue: {len(scenes)} scenes; model={args.model}; effort={args.effort}",flush=True)
    if execute:
        session.run()
    else:
        print("Plan only: no model calls, camera/serial connections, or motor writes.",flush=True)
    print(f"Session status: {data['status']}\nSpreadsheet: {output/'evaluation.xlsx'}",flush=True)
    return output


if __name__=="__main__":
    main()
