using DreamHopper.IO;
using Rhino.Geometry;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DreamHopper.ViewModels
{
    public static class MeshOperations
    {
        public static Point3d GetVertex(List<double> list)
        {
            return new Point3d(list[0], list[1], list[2]);
        }

        public static Color GetColor(List<double> list)
        {
            var mappedValues = MapValues(list, 0, 1, 0, 255);
            return Color.FromArgb((int)mappedValues[0], (int)mappedValues[1], (int)mappedValues[2]);
        }

        public static List<double> MapValues(List<double> values, double fromMin, double fromMax, double toMin, double toMax)
        {
            var mappedValues = new List<double>();
            foreach (double value in values)
            {
                var fromAbs = value - fromMin;
                var fromMaxAbs = fromMax - fromMin;

                var normal = fromAbs / fromMaxAbs;

                var toMaxAbs = toMax - toMin;
                var toAbs = toMaxAbs * normal;

                var to = toAbs + toMin;

                mappedValues.Add((int)Math.Floor(to));
            }
            return mappedValues;
        }

        public static Mesh CreateMesh(this DreamHopperMesh dmesh)
        {
            // Split the list
            var verticesData = dmesh.Vertices.Select(x => x.ChunkBy(3)[0]);
            var colorData = dmesh.Vertices.Select(x => x.ChunkBy(3)[1]);

            // Create the mesh
            var mesh = new Mesh();

            // Create the vertices
            var vertices = verticesData.Select(x => GetVertex(x));
            mesh.Vertices.AddVertices(vertices);

            for (int i = 0; i < dmesh.Faces.Count; i++)
            {
                List<int> protoface = dmesh.Faces[i];

                MeshFace rhinoFace = new MeshFace(protoface[0], protoface[1], protoface[2]);
                mesh.Faces.AddFace(rhinoFace);
            }

            // Create the vertex color
            var colors = colorData.Select(x => GetColor(x));
            mesh.VertexColors.SetColors(colors.ToArray());

            // Output
            return mesh;
        }

        public static List<List<T>> ChunkBy<T>(this List<T> source, int chunkSize)
        {
            return source
                .Select((x, i) => new { Index = i, Value = x })
                .GroupBy(x => x.Index / chunkSize)
                .Select(x => x.Select(v => v.Value).ToList())
                .ToList();
        }


    }
}