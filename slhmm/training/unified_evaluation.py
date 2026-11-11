"""Common full-edge metrics, plus separately labelled cropped-reference diagnostics."""
from __future__ import annotations
import numpy as np
import torch
from shapely.ops import substring
from slhmm.training.evaluation import FullViterbiEvaluator, _sequence_lcs_f1, _road_length_metrics, _geometry_length_metrics
from slhmm.training.lattice import score_lattice
from slhmm.training.common import write_json
from slhmm.training.trainer import _autocast


def cropped_length_metrics(predicted, truth, lengths, entry, exit):
    """Occurrence-weighted LCS against provided clipped truth intervals.

Predictions stay full-edge; no true offsets are ever fed to the decoder.
The max recurrence also handles different lengths for repeated occurrences.
"""
    truth_weights = [max(0., min(lengths[s], b)-max(0., a)) for s,a,b in zip(truth,entry,exit)]
    row=[0.]*(len(truth)+1)
    for state in predicted:
        prior=0.
        for j,target in enumerate(truth,1):
            saved=row[j]
            row[j]=max(row[j],row[j-1],prior+truth_weights[j-1] if state==target else 0.)
            prior=saved
    correct=row[-1]; pl=sum(lengths[s] for s in predicted); tl=sum(truth_weights)
    p=correct/pl if pl else 0.; r=correct/tl if tl else 0.
    return dict(correct=correct,predicted=pl,truth=tl,precision=p,recall=r,f1=2*p*r/(p+r) if p+r else 0.,
                rmf=(pl+tl-2*correct)/tl if tl else 0.)


class UnifiedEvaluator(FullViterbiEvaluator):
    def __init__(self,engine):
        super().__init__(engine)
        expected=list(engine.resources.alignment_val)
        if engine.config.limit_trajectories is not None: expected=expected[:engine.config.limit_trajectories]
        absent=set(expected)-self.candidates.keys()
        if absent: raise ValueError(f"Validation candidate pools missing {len(absent)} trajectories")

    def decode(self,trajectory_id,alignment):
        engine=self.engine
        if engine.config.variant=="baseline":
            result=super().decode(trajectory_id,alignment)
        else:
            pool=self.candidates[trajectory_id]
            with _autocast(engine.device,engine.amp):
                lattice=score_lattice(engine,[(trajectory_id,alignment)],[pool])[0]
            values=lattice["emit"][0]; pointers=[]
            for t,transition in enumerate(lattice["trans"],1):
                best,ptr=(values[:,None]+transition).max(0)
                values=best+lattice["emit"][t]; pointers.append(ptr)
            if not torch.isfinite(values).any():
                result=self._failed_decode_result(trajectory_id,alignment,"no_complete_route_after_unbounded_fallback")
            else:
                selected=[int(values.argmax())]
                for ptr in reversed(pointers): selected.append(int(ptr[selected[-1]]))
                selected.reverse()
                anchors=[pool[t][j] for t,j in enumerate(selected)]
                expanded=[anchors[0]]; ratios=[]
                for t,(left,right) in enumerate(zip(anchors,anchors[1:])):
                    path=engine.router.path_states(left,right)
                    if path is None: raise AssertionError("Scored reachable pair cannot be expanded")
                    expanded.extend(path[1:])
                    # Every expanded path must reproduce the features scored by
                    # training, including the same shortest-path tie handling.
                    row=selected[t]*len(pool[t+1])+selected[t+1]
                    expected=lattice["routes"]["features"][t][row]
                    actual=engine.router.features_for_path(path)
                    if not np.allclose(actual,expected,rtol=2e-5,atol=1e-3):
                        raise AssertionError("Routing feature/path inconsistency")
                    if lattice["dnet"][t] is not None:
                        ratios.append(float(lattice["dnet"][t][row])/float(lattice["routes"]["drefs"][t]))
                edges=[self.state_edges[s] for s in expanded]
                truth=list(alignment["truth_state_ids"])
                predicted_lines=self.matcher.route_geometries_metric(edges)
                truth_lines=self.matcher.route_geometries_metric([self.state_edges[s] for s in truth])
                road=_road_length_metrics(expanded,truth,self.state_lengths_m)
                geom=_geometry_length_metrics(predicted_lines,truth_lines,self.config.geometry_tolerance_m)
                corridor=_geometry_length_metrics(predicted_lines,truth_lines,50.)
                lcs=_sequence_lcs_f1(expanded,truth)
                continuity=float(np.mean([a[1]==b[0] for a,b in zip(edges,edges[1:])])) if len(edges)>1 else 1.
                result=dict(trajectory_id=trajectory_id,success=True,selected_state_ids=anchors,expanded_state_ids=expanded,
                            directed_edge_lcs_f1=lcs,geometry_length_f1=geom["f1"],**road,
                            cmf_50=1-corridor["recall"],corridor_uncovered_length_50m=corridor["truth_uncovered_length_m"],
                            corridor_truth_length_m=corridor["truth_length_m"],topology_continuity=continuity,
                            composite=.5*lcs+.5*geom["f1"],detour_ratios=ratios)
            result["fallback_unbounded_steps"]=lattice["routes"]["fallback_unbounded_steps"]
            result["transition_steps"]=max(0,len(pool)-1)
        self._cropped(result,alignment)
        return result

    def _cropped(self,result,alignment):
        if "road_entry_offsets_m" not in alignment: return
        truth=list(alignment["truth_state_ids"])
        predicted=result.get("expanded_state_ids",[])
        metrics=cropped_length_metrics(predicted,truth,self.state_lengths_m,alignment["road_entry_offsets_m"],alignment["road_exit_offsets_m"])
        clipped=[]
        for state,a,b in zip(truth,alignment["road_entry_offsets_m"],alignment["road_exit_offsets_m"]):
            line=self.matcher.edge_geometries[self.matcher._edge_to_index[self.state_edges[state]]]
            part=substring(line,max(0.,a),min(line.length,b))
            if part.geom_type=="LineString" and part.length>0: clipped.append(part)
        predicted_lines=self.matcher.route_geometries_metric([self.state_edges[s] for s in predicted])
        corridor=_geometry_length_metrics(predicted_lines,clipped,50.)
        metrics.update(corridor_truth=corridor["truth_length_m"],corridor_missing=corridor["truth_uncovered_length_m"])
        result["cropped_reference_diagnostic"]=metrics

    def evaluate(self,output_path=None,**kwargs):
        report=super().evaluate(None,**kwargs)
        metrics=report["metrics"]; trajectories=report["trajectories"]
        metrics["metric_protocol"]="full_edge_wlcs_micro"
        metrics["dataset"]=self.config.dataset; metrics["variant"]=self.config.variant; metrics["seed"]=self.config.seed
        if any("fallback_unbounded_steps" in r for r in trajectories):
            steps=sum(r.get("transition_steps",0) for r in trajectories)
            fallback=sum(r.get("fallback_unbounded_steps",0) for r in trajectories)
            metrics["fallback_unbounded_steps"]=fallback
            metrics["fallback_unbounded_ratio"]=fallback/max(1,steps)
            ratios=[v for r in trajectories for v in r.get("detour_ratios",[])]
            metrics["selected_detour_ratio_above_3"]=float(np.mean(np.asarray(ratios)>3)) if ratios else None
        cropped=[r["cropped_reference_diagnostic"] for r in trajectories if "cropped_reference_diagnostic" in r]
        if cropped:
            sums={k:sum(r[k] for r in cropped) for k in ("correct","predicted","truth","corridor_truth","corridor_missing")}
            p=sums["correct"]/max(sums["predicted"],1e-12); r=sums["correct"]/max(sums["truth"],1e-12)
            report["cropped_reference_diagnostics"]=dict(protocol="full_prediction_vs_supplied_clipped_truth",precision_micro=p,recall_micro=r,
                f1_micro=2*p*r/(p+r) if p+r else 0.,f1_macro=float(np.mean([m["f1"] for m in cropped])),
                rmf=(sums["predicted"]+sums["truth"]-2*sums["correct"])/max(sums["truth"],1e-12),
                cmf_50=sums["corridor_missing"]/max(sums["corridor_truth"],1e-12))
        if output_path: write_json(output_path,report)
        return report
