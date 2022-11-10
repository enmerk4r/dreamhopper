using DreamHopper.MVVM;
using System.Windows.Input;

namespace DreamHopper.ViewModels.ViewModels
{
    public class TextInputViewModel : BaseInputViewModel
    {
        public ICommand ClearTextCommand { get; set; }

        public TextInputViewModel(string name, string jsonPropertyName)
            : base(name, jsonPropertyName, string.Empty)
        {
            this.ClearTextCommand = new RelayCommand(this.ClearText);
        }

        public void ClearText()
        {
            this.Value = string.Empty;
        }
    }
}