using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DreamHopper.IO
{
    public class SubmissionReceipt
    {
        [JsonProperty("id")]
        public string Id { get; set; }

        public SubmissionReceipt()
        {

        }

        public SubmissionReceipt(string id)
        {
            Id = id;
        }
    }
}
