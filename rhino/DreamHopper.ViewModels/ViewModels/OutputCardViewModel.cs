using DreamHopper.MVVM;
using System.Drawing;

namespace DreamHopper.ViewModels.ViewModels
{
    public class OutputCardViewModel : BaseViewModel
    {
        public bool IsSelected { get; set; }
        public Bitmap Image { get; set; }

        public OutputCardViewModel(Bitmap image, bool isSelected = true)
        {
            this.Image = image;
            this.IsSelected = isSelected;
        }
    }
}

