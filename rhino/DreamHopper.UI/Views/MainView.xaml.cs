using DreamHopper.ViewModels;
using Rhino;
using System.Windows.Controls;

namespace DreamHopper.UI.Views
{
    public partial class MainView : UserControl
    {
        public MainView(RhinoDoc doc)
        {
            InitializeComponent();
            this.DataContext = new DreamHopperCoreViewModel(doc.Name);
        }
    }
}
