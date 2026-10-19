// Compile/transport harness only. Does not substitute for testing Unity rendering.
using System;
using System.Text.Json;
namespace UnityEngine {
    public class Object { public static implicit operator bool(Object o)=>o!=null; public static void Destroy(Object o){} }
    public class MonoBehaviour:Object {}
    public class TooltipAttribute:Attribute { public TooltipAttribute(string s){} }
    public class ContextMenu:Attribute { public ContextMenu(string s){} }
    public class Transform:Object { public Vector3 TransformPoint(Vector3 p)=>p; public Vector3 InverseTransformPoint(Vector3 p)=>p; }
    public struct Vector3 { public float x,y,z; public Vector3(float x,float y,float z){this.x=x;this.y=y;this.z=z;} }
    public struct Matrix4x4 { public static Matrix4x4 Scale(Vector3 p)=>new Matrix4x4(); public static Matrix4x4 operator*(Matrix4x4 a,Matrix4x4 b)=>a; }
    public struct Color { public static Color cyan,green; }
    public static class Gizmos { public static Color color; public static void DrawSphere(Vector3 p,float r){} public static void DrawLine(Vector3 a,Vector3 b){} }
    public class TextAsset { public string text; }
    public static class Time { public static double realtimeSinceStartupAsDouble => (double)System.Diagnostics.Stopwatch.GetTimestamp()/System.Diagnostics.Stopwatch.Frequency; }
    public static class JsonUtility { public static T FromJson<T>(string s)=>JsonSerializer.Deserialize<T>(s,new JsonSerializerOptions{IncludeFields=true}); }
    public class Mesh:Object {
        public string name; public Vector3[] vertices=Array.Empty<Vector3>(); public int[] triangles=Array.Empty<int>();
        public void Clear(){vertices=Array.Empty<Vector3>();triangles=Array.Empty<int>();}
        public void RecalculateNormals(){} public void RecalculateBounds(){}
    }
    public class MeshFilter:Object { public Transform transform=new Transform(); public Mesh sharedMesh; }
}
