using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Configuration;
using DreamHopper.IO;

namespace DreamHopper.WebClient
{
    public class InferenceClient
    {
        private Configuration _configuration;
        private HttpClient _client;

        private int _checkFrequency;
        public int CheckFrequency
        {
            get
            {
                if (_checkFrequency == 0)
                {
                    _checkFrequency = int.Parse(GetAppSetting("checkFrequency"));
                }
                return _checkFrequency;
            }
        }

        private string _server;
        public string Server
        {
            get
            {
                if (string.IsNullOrEmpty(_server))
                {
                    _server = GetAppSetting("server");
                }
                return _server;
            }
        }

        private string _submitUrl;
        public string SubmitUrl
        {
            get
            {
                if (string.IsNullOrEmpty(_submitUrl))
                {
                    Uri baseUri = new Uri(this.Server);
                    Uri submitUri = new Uri(baseUri, "generate");
                    _submitUrl = submitUri.ToString();
                }
                return _submitUrl;
            }
        }


        private string _checkUrl;
        public string CheckUrl
        {
            get
            {
                if (string.IsNullOrEmpty(_checkUrl))
                {
                    Uri baseUri = new Uri(this.Server);
                    Uri checkUri = new Uri(baseUri, "check");
                    _checkUrl = checkUri.ToString();
                }
                return _checkUrl;
            }
        }

        public InferenceClient()
        {
            this._client = new HttpClient();
            string exeConfigPath = typeof(InferenceClient).Assembly.Location;
            try
            {
                _configuration = ConfigurationManager.OpenExeConfiguration(exeConfigPath);
            }
            catch (Exception ex)
            {
                throw new System.IO.FileNotFoundException("Couldn't locate the Web.config file", ex);
            }
        }

        private string GetAppSetting(string key)
        {
            if (_configuration == null) throw new Exception("Failed to load configuration file");
            KeyValueConfigurationElement element = _configuration.AppSettings.Settings[key];
            if (element != null)
            {
                string value = element.Value;
                if (!string.IsNullOrEmpty(value))
                    return value;
            }
            return string.Empty;
        }

        public async Task<SubmissionReceipt> SubmitRequest(DreamHopperDTO req)
        {
            string content = JsonConvert.SerializeObject(req);
            StringContent payload = new StringContent(content);
            payload.Headers.ContentType = MediaTypeHeaderValue.Parse("application/json");
            var response = await this._client.PostAsync(this.SubmitUrl, payload);
            if (response.StatusCode != System.Net.HttpStatusCode.OK)
            {
                throw new Exception($"Http Error: {(int)response.StatusCode} | {response.ReasonPhrase}");
            }
            var res = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<SubmissionReceipt>(res);
        }

        public async Task<DreamHopperDTO> CheckRequestStatus(SubmissionReceipt receipt)
        {
            StringContent payload = new StringContent(JsonConvert.SerializeObject(receipt));
            payload.Headers.ContentType = MediaTypeHeaderValue.Parse("application/json");
            var response = await this._client.PostAsync(this.CheckUrl, payload);
            if (response.StatusCode != System.Net.HttpStatusCode.OK)
            {
                throw new Exception($"Http Error: {(int)response.StatusCode} | {response.ReasonPhrase}");
            }
            var res = await response.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<DreamHopperDTO>(res);
        }
    }
}
