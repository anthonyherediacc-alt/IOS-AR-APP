import test from 'node:test';
import assert from 'node:assert/strict';
import {evaluate} from './evaluate.mjs';
const frame = {sessionID:'synthetic',worldOriginEpoch:1,frameID:1,state:'tracking',inferenceMS:5,
 observationAge:0,joints:[{name:'leftKnee',image:[0.2,0.3]}]};
test('synthetic landmark error and positive availability',()=>{
 const r=evaluate([frame],[{...frame,targetPresent:true,joints:{leftKnee:[0.2,0.4]}}]);
 assert.equal(r.selectedTargetAvailability,1); assert.ok(Math.abs(r.meanLandmarkErrorNormalizedUV-0.1)<1e-9);
});
test('negative false detection',()=>{
 assert.equal(evaluate([frame],[{...frame,targetPresent:false,joints:{}}]).falsePositives,1);
});
test('missing frame rejected, never scored as zero error',()=>{
 assert.throws(()=>evaluate([],[{...frame,targetPresent:true}]),/Missing/);
});
test('missing landmark remains unavailable',()=>{
 const r=evaluate([{...frame,joints:[],state:'lost'}],[{...frame,targetPresent:true,joints:{leftKnee:[0,0]}}]);
 assert.equal(r.selectedTargetAvailability,0);assert.equal(r.meanLandmarkErrorNormalizedUV,null);
});
test('epoch separates repeated frame ids',()=>{
 assert.throws(()=>evaluate([frame],[{...frame,worldOriginEpoch:2}]),/Missing/);
});
