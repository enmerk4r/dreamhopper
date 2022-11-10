using System.Collections.Generic;
using System.Collections.ObjectModel;

namespace DreamHopper.ViewModels.ViewModels
{
    public class DropdownViewModel : BaseInputViewModel
    {
        public ObservableCollection<KeyValuePair<string, object>> Options { get; set; }
        public KeyValuePair<string, object> SelectedOption { get; set; }

        public DropdownViewModel(string name, string jsonPropertyName, List<KeyValuePair<string, object>> options, int selectedIndex)
            : base(name, jsonPropertyName, options[selectedIndex].Value)
        {
            this.Options = new ObservableCollection<KeyValuePair<string, object>>(options);
            this.SelectedOption = this.Options[selectedIndex];
            this.PropertyChanged += DropdownViewModel_PropertyChanged;

        }

        private void DropdownViewModel_PropertyChanged(object sender, System.ComponentModel.PropertyChangedEventArgs e)
        {
            if (e.PropertyName == nameof(this.SelectedOption))
            {
                this.Value = this.SelectedOption.Value;
            }
        }
    }
}
