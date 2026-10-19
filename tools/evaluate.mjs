import fs from 'node:fs';
import {pathToFileURL} from 'node:url';

// Annotation coordinates are raw sensor top-left normalized UV, never screen pixels.
export function evaluate(packets, annotations) {
  const lookup = new Map(packets.map(p => [`${p.sessionID}:${p.worldOriginEpoch}:${p.frameID}`, p]));
  let positives=0, detected=0, negatives=0, falsePositives=0, matched=0;
  const errors=[], timings=[], ages=[];
  for (const a of annotations) {
    if(typeof a.targetPresent!=='boolean') throw new Error('targetPresent must be an explicit Boolean');
    const p=lookup.get(`${a.sessionID}:${a.worldOriginEpoch}:${a.frameID}`);
    if (!p) throw new Error(`Missing annotated frame ${a.frameID}`);
    const present=p.state==='tracking' && p.joints.some(j=>j.image || j.world);
    if (a.targetPresent) { positives++; if(present) detected++; }
    else { negatives++; if(present) falsePositives++; }
    for(const [name, uv] of Object.entries(a.joints ?? {})) {
      if(!Array.isArray(uv) || uv.length!==2 || !uv.every(x=>Number.isFinite(x) && x>=0 && x<=1))
        throw new Error(`Invalid annotation for ${name}`);
      const j=p.joints.find(j=>j.name===name);
      if(j?.image) { errors.push(Math.hypot(j.image[0]-uv[0],j.image[1]-uv[1])); matched++; }
    }
    if(Number.isFinite(p.inferenceMS)) timings.push(p.inferenceMS);
    if(Number.isFinite(p.observationAge)) ages.push(p.observationAge);
  }
  const mean = a=>a.length ? a.reduce((s,x)=>s+x,0)/a.length : null;
  return {annotatedFrames: annotations.length, positives, detected, negatives, falsePositives,
    selectedTargetAvailability: positives ? detected/positives : null, matchedLandmarks:matched,
    meanLandmarkErrorNormalizedUV:mean(errors),meanInferenceMS:mean(timings),meanObservationAgeSeconds:mean(ages),
    note:'Selected-target output evaluation; acquisition/detector recall and physical identity require separate annotation. No metric accuracy inferred.'};
}
if (process.argv[1] && import.meta.url===pathToFileURL(process.argv[1]).href) {
  const read=p=>fs.readFileSync(p,'utf8').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);
  console.log(JSON.stringify(evaluate(read(process.argv[2]),read(process.argv[3])),null,2));
}
