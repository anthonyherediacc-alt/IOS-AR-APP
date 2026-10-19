using System;
using System.Collections.Generic;
using UnityEngine;

// No rig or AR package required. Attach to an empty GameObject; assign a JSONL renamed .txt.
public sealed class PoseReplay : MonoBehaviour
{
    [Serializable] public class Joint { public string name, parentName; public float[] world; public string provenance; }
    [Serializable] public class Packet {
        public int schemaVersion, frameID, worldOriginEpoch;
        public string sessionID, targetID, state;
        public double timestamp;
        public Joint[] joints;
    }
    public TextAsset recording;
    [Tooltip("Explicit alignment from converted phone world to this Unity scene; identity is replay-only.")]
    public Transform sharedSpace;
    public float pointRadius = 0.025f;
    private readonly List<Packet> packets = new List<Packet>();
    private int index;
    private double started;
    private bool playing;
    private readonly Dictionary<string, Vector3> points = new Dictionary<string, Vector3>();
    private readonly List<KeyValuePair<string,string>> hierarchy = new List<KeyValuePair<string,string>>();
    private static readonly string[,] bones = {
        {"neck","leftShoulder"},{"neck","rightShoulder"},{"leftShoulder","leftElbow"},
        {"leftElbow","leftWrist"},{"rightShoulder","rightElbow"},{"rightElbow","rightWrist"},
        {"leftShoulder","leftHip"},{"rightShoulder","rightHip"},{"leftHip","rightHip"},
        {"leftHip","leftKnee"},{"leftKnee","leftAnkle"},{"rightHip","rightKnee"},{"rightKnee","rightAnkle"}
    };
    public static Matrix4x4 ConvertPose(Matrix4x4 arkit) {
        var basis = Matrix4x4.Scale(new Vector3(1,1,-1));
        return basis * arkit * basis;
    }
    private void Start() {
        if (recording == null) return;
        foreach (var line in recording.text.Split('\n')) {
            if (string.IsNullOrWhiteSpace(line)) continue;
            var p = JsonUtility.FromJson<Packet>(line);
            if(p == null || p.schemaVersion != 1 || p.joints == null || double.IsNaN(p.timestamp) || double.IsInfinity(p.timestamp))
                throw new FormatException("Unsupported or invalid pose packet");
            if(packets.Count > 0) {
                var prior=packets[packets.Count-1];
                if(p.sessionID != prior.sessionID || p.timestamp < prior.timestamp)
                    throw new FormatException("Replay requires one monotonic session");
            }
            packets.Add(p);
        }
        started=Time.realtimeSinceStartupAsDouble; playing=packets.Count>0;
    }
    private void Update() {
        if(!playing) return;
        var time=packets[0].timestamp+Time.realtimeSinceStartupAsDouble-started;
        while(index+1<packets.Count && packets[index+1].timestamp<=time) index++;
        var p=packets[index]; points.Clear(); hierarchy.Clear();
        // Hide gaps and end-of-recording data; do not make a frozen skeleton appear live.
        if(time-p.timestamp>0.25 || p.state!="tracking") return;
        foreach(var j in p.joints) {
            if(j.world==null || j.world.Length!=3 || Array.Exists(j.world,x=>float.IsNaN(x)||float.IsInfinity(x))) continue;
            var v=new Vector3(j.world[0],j.world[1],-j.world[2]);
            points[j.name]=sharedSpace ? sharedSpace.TransformPoint(v) : v;
            if(!string.IsNullOrEmpty(j.parentName)) hierarchy.Add(new KeyValuePair<string,string>(j.parentName,j.name));
        }
    }
    private void OnDrawGizmos() {
        Gizmos.color=Color.cyan;
        foreach(var p in points.Values) Gizmos.DrawSphere(p,pointRadius);
        foreach(var edge in hierarchy)
            if(points.TryGetValue(edge.Key,out var parent) && points.TryGetValue(edge.Value,out var child)) Gizmos.DrawLine(parent,child);
        for(var i=0;i<bones.GetLength(0);i++)
            if(points.TryGetValue(bones[i,0],out var a) && points.TryGetValue(bones[i,1],out var b)) Gizmos.DrawLine(a,b);
    }
}
