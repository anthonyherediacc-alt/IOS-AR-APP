using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

// Desktop Unity. USB: run iproxy 8765:8765, then connect to 127.0.0.1.
// Wi-Fi: set host to the phone's IP. Phone Start Unity stream must be active.
public sealed class PoseLiveReceiver : MonoBehaviour
{
    [Serializable] public class SurfacePoint { public float x, y, z; }
    [Serializable] public class SurfaceMesh {
        public SurfacePoint[] vertices;
        public int[] triangles;
        public string status, targetID, representation;
    }
    [Serializable] public class Frame {
        public int protocolVersion;
        public PoseReplay.Packet packet;
        public SurfacePoint[] surface;
        public SurfaceMesh mesh;
    }
    public string host = "127.0.0.1";
    public int port = 8765;
    public string pairingCode = "";
    public Transform sharedSpace;
    [Tooltip("Assign a MeshFilter with a MeshRenderer and material to display the measured surface.")]
    public MeshFilter surfaceOutput;
    public string status = "Disconnected";
    public Frame Latest { get; private set; }
    private CancellationTokenSource cancellation;
    private TcpClient client;
    private readonly object sync = new object();
    private string pending, workerStatus;
    private double received, lastReceived;
    private readonly List<Vector3> joints = new List<Vector3>();
    private readonly List<Vector3> surface = new List<Vector3>();
    private Mesh renderedMesh;
    private static double Clock => (double)System.Diagnostics.Stopwatch.GetTimestamp() / System.Diagnostics.Stopwatch.Frequency;

    [ContextMenu("Connect stream")]
    public void Connect() {
        Disconnect();
        if (string.IsNullOrWhiteSpace(pairingCode) || pairingCode.Length != 6) { status="Enter phone pairing code"; return; }
        cancellation = new CancellationTokenSource();
        var token = cancellation.Token;
        client = new TcpClient(); var socket = client;
        var address = host; var tcpPort = port; var code = pairingCode;
        status = "Connecting";
        _ = Task.Run(async () => {
            try {
                using (token.Register(() => socket.Close())) {
                    await socket.ConnectAsync(address,tcpPort);
                    var stream = socket.GetStream();
                    var auth = Encoding.UTF8.GetBytes(code+"\n");
                    await stream.WriteAsync(auth,0,auth.Length,token);
                    var buffer = new byte[8192];
                    using (var line = new MemoryStream()) {
                        while (!token.IsCancellationRequested) {
                            var count = await stream.ReadAsync(buffer,0,buffer.Length,token);
                            if (count == 0) throw new IOException("Phone disconnected or pairing code rejected");
                            for (int i=0;i<count;i++) {
                                if (buffer[i]==10) {
                                    var json = Encoding.UTF8.GetString(line.ToArray()); line.SetLength(0);
                                    lock(sync) { if(!token.IsCancellationRequested) { pending=json; received=Clock; } }
                                    var ack=Encoding.UTF8.GetBytes("ACK\n");
                                    await stream.WriteAsync(ack,0,ack.Length,token);
                                } else {
                                    if(line.Length>=1048576) throw new IOException("Oversized frame rejected");
                                    line.WriteByte(buffer[i]);
                                }
                            }
                        }
                    }
                }
            } catch(Exception e) {
                lock(sync) { if(!token.IsCancellationRequested) workerStatus=e.Message; }
            } finally { socket.Close(); }
        },token);
    }
    [ContextMenu("Disconnect stream")]
    public void Disconnect() {
        cancellation?.Cancel(); cancellation?.Dispose(); cancellation=null;
        client?.Close(); client=null;
        lock(sync) { pending=null; workerStatus=null; }
        Latest=null; joints.Clear(); surface.Clear(); status="Disconnected";
        ClearMesh();
    }
    private void OnDisable() { Disconnect(); }
    private void OnDestroy() { if(renderedMesh) Destroy(renderedMesh); }
    private void ClearMesh() { if(renderedMesh) renderedMesh.Clear(); }
    private void RenderMesh(SurfaceMesh data) {
        ClearMesh();
        if(data==null || data.status!="tracking" || surfaceOutput==null) return;
        if(data.vertices==null || data.triangles==null || data.vertices.Length>3072 ||
            data.triangles.Length>18000 || data.triangles.Length%3!=0)
            throw new FormatException("Invalid mesh sizes");
        var vertices=new Vector3[data.vertices.Length];
        for(int i=0;i<vertices.Length;i++) {
            var p=data.vertices[i];
            if(p==null || !Finite(p.x) || !Finite(p.y) || !Finite(p.z)) throw new FormatException("Invalid vertex");
            vertices[i]=surfaceOutput.transform.InverseTransformPoint(Convert(p.x,p.y,p.z));
        }
        var triangles=(int[])data.triangles.Clone();
        foreach(var i in triangles) if(i<0 || i>=vertices.Length) throw new FormatException("Invalid mesh index");
        // Reflecting ARKit Z into Unity changes handedness; reverse triangle winding.
        for(int i=0;i<triangles.Length;i+=3) { var t=triangles[i+1]; triangles[i+1]=triangles[i+2]; triangles[i+2]=t; }
        if(!renderedMesh) { renderedMesh=new Mesh(); renderedMesh.name="Live LiDAR target surface"; }
        renderedMesh.vertices=vertices; renderedMesh.triangles=triangles;
        renderedMesh.RecalculateNormals(); renderedMesh.RecalculateBounds(); surfaceOutput.sharedMesh=renderedMesh;
    }
    private static bool Finite(float f) => !float.IsNaN(f) && !float.IsInfinity(f);
    private Vector3 Convert(float x,float y,float z) {
        var p=new Vector3(x,y,-z);
        return sharedSpace ? sharedSpace.TransformPoint(p) : p;
    }
    private void Update() {
        string json,error; double arrival;
        lock(sync) { json=pending; pending=null; arrival=received; error=workerStatus; workerStatus=null; }
        if(error!=null) { status=error; Latest=null; joints.Clear(); surface.Clear(); ClearMesh(); return; }
        if(json!=null) {
            try {
                var f=JsonUtility.FromJson<Frame>(json);
                if(f==null || f.protocolVersion!=1 || f.packet==null || f.packet.schemaVersion!=1 ||
                    f.packet.joints==null || double.IsNaN(f.packet.timestamp) || double.IsInfinity(f.packet.timestamp))
                    throw new FormatException("Invalid/unsupported stream frame");
                Latest=f; lastReceived=arrival; joints.Clear(); surface.Clear();
                RenderMesh(f.mesh);
                if(f.packet.state=="tracking") foreach(var j in f.packet.joints) {
                    var w=j.world;
                    if(w!=null && w.Length==3 && Array.TrueForAll(w,Finite)) joints.Add(Convert(w[0],w[1],w[2]));
                }
                if(f.surface!=null && f.surface.Length<=320) foreach(var p in f.surface)
                    if(p!=null && Finite(p.x) && Finite(p.y) && Finite(p.z)) surface.Add(Convert(p.x,p.y,p.z));
                status="Receiving: "+f.packet.state+" • origin "+f.packet.worldOriginEpoch;
            } catch(Exception e) { status=e.Message; Latest=null; joints.Clear(); surface.Clear(); ClearMesh(); }
        }
        if(Latest!=null && Clock-lastReceived>0.5) {
            Latest=null; joints.Clear(); surface.Clear(); ClearMesh(); status="Stale stream — waiting for fresh frames";
        }
    }
    private void OnDrawGizmos() {
        Gizmos.color=Color.cyan; foreach(var p in joints) Gizmos.DrawSphere(p,0.025f);
        Gizmos.color=Color.green; foreach(var p in surface) Gizmos.DrawSphere(p,0.008f);
    }
}
