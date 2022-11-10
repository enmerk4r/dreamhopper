using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DreamHopper.IO
{
    public class DreamHopperDTO
    {
        [JsonProperty("text")]
        public string Prompt { get; set; }

        [JsonProperty("user")]
        public string User { get; set; }

        [JsonProperty("iters")]
        public int Iterations { get; set; }

        [JsonProperty("w")]
        public int ClipWidth { get; set; }

        [JsonProperty("h")]
        public int ClipHeight { get; set; }

        [JsonProperty("W")]
        public int RenderWidth { get; set; }

        [JsonProperty("H")]
        public int RenderHeight { get; set; }

        [JsonProperty("threshold")]
        public double Threshold { get; set; }

        [JsonProperty("seed")]
        public int Seed { get; set; }

        [JsonProperty("done")]
        public bool Done { get; set; }

        [JsonProperty("mesh")]
        public DreamHopperMesh Mesh { get; set; }

        public DreamHopperDTO(string prompt, string user, int iterations, int clip, double threshold, int seed)
        {
            Prompt = prompt;
            User = user;
            Iterations = iterations;
            ClipWidth = clip;
            ClipHeight = clip;
            RenderWidth = clip;
            RenderHeight = clip;
            Threshold = threshold;
            Seed = seed;
            Done = false;
        }

        public DreamHopperDTO()
        {

        }
    }
}
