using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

class Program {
    static void Check(bool condition,string message) { if(!condition) throw new Exception(message); }
    static void Tick(PoseLiveReceiver receiver) => typeof(PoseLiveReceiver).GetMethod("Update",BindingFlags.Instance|BindingFlags.NonPublic).Invoke(receiver,null);
    static async Task Until(PoseLiveReceiver receiver,Func<bool> predicate) {
        for(int i=0;i<100;i++) { Tick(receiver); if(predicate()) return; await Task.Delay(20); }
        throw new Exception("Timed out: "+receiver.status);
    }
    static async Task Send(NetworkStream stream,string line) {
        var data=Encoding.UTF8.GetBytes(line+"\n");
        for(int i=0;i<data.Length;i+=17) await stream.WriteAsync(data,i,Math.Min(17,data.Length-i));
    }
    static async Task Main() {
        var listener=new TcpListener(IPAddress.Loopback,0); listener.Start();
        var receiver=new PoseLiveReceiver { host="127.0.0.1",port=((IPEndPoint)listener.LocalEndpoint).Port,pairingCode="123456",surfaceOutput=new MeshFilter() };
        try {
            receiver.Connect();
            using var peer=await listener.AcceptTcpClientAsync().WaitAsync(TimeSpan.FromSeconds(5));
            var stream=peer.GetStream(); using var reader=new StreamReader(stream,Encoding.UTF8,false,1024,true);
            Check(await reader.ReadLineAsync().WaitAsync(TimeSpan.FromSeconds(5))=="123456","Pairing code not sent");
            const string frame="""{"protocolVersion":1,"packet":{"schemaVersion":1,"timestamp":1,"state":"tracking","joints":[]},"mesh":{"status":"tracking","vertices":[{"x":0,"y":0,"z":-1},{"x":1,"y":0,"z":-1},{"x":0,"y":1,"z":-1}],"triangles":[0,1,2]}}""";
            await Send(stream,frame);
            Check(await reader.ReadLineAsync().WaitAsync(TimeSpan.FromSeconds(5))=="ACK","Missing frame ACK");
            await Until(receiver,()=>receiver.surfaceOutput.sharedMesh?.triangles.Length==3);
            Check(receiver.surfaceOutput.sharedMesh.vertices[0].z==1,"ARKit Z reflection failed");
            Check(receiver.surfaceOutput.sharedMesh.triangles[1]==2,"Reflected winding not reversed");
            await Send(stream,frame.Replace("[0,1,2]","[0,1,99]"));
            await reader.ReadLineAsync().WaitAsync(TimeSpan.FromSeconds(5));
            await Until(receiver,()=>receiver.status.Contains("Invalid mesh index"));
            Check(receiver.surfaceOutput.sharedMesh.triangles.Length==0,"Invalid mesh retained geometry");
            await Send(stream,frame); await reader.ReadLineAsync().WaitAsync(TimeSpan.FromSeconds(5));
            await Until(receiver,()=>receiver.Latest!=null);
            await Task.Delay(600); Tick(receiver);
            Check(receiver.Latest==null && receiver.surfaceOutput.sharedMesh.triangles.Length==0,"Stale geometry was not cleared");
            Console.WriteLine("PASS: Unity sample compiles; TCP fragmentation, pairing, ACK, mesh coordinate conversion, invalid-index rejection and stale clearing.");
        } finally { receiver.Disconnect(); listener.Stop(); }
    }
}
