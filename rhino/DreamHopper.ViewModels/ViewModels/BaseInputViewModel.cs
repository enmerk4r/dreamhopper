using DreamHopper.MVVM;
using System.Collections.Generic;

namespace DreamHopper.ViewModels.ViewModels
{
    public class BaseInputViewModel : BaseViewModel
    {
        public string Name { get; set; }
        public string JsonPropertyName { get; set; }
        public object Value { get; set; }

        public KeyValuePair<string, object> GetValuePair()
        {
            return new KeyValuePair<string, object>(this.JsonPropertyName, this.Value);
        }

        public BaseInputViewModel(string name, string jsonPropertyName, object value)
        {
            this.Name = name;
            this.JsonPropertyName = jsonPropertyName;
            this.Value = value;
        }
    }
}
