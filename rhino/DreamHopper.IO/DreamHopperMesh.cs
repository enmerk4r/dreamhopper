using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DreamHopper.IO
{
    public class DreamHopperMesh
    {
        [JsonProperty("vertices")]
        public List<List<double>> Vertices { get; set; }

        [JsonProperty("faces")]
        public List<List<int>> Faces { get; set; }

        public bool IsValid => Vertices != null && Faces != null;
    }
}
