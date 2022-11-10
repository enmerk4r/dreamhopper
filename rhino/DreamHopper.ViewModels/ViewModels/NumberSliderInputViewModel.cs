namespace DreamHopper.ViewModels.ViewModels
{
    public class NumberSliderInputViewModel : BaseInputViewModel
    {
        public double Minimum { get; set; }
        public double Maximum { get; set; }
        public double Step { get; set; }
        public NumberSliderInputViewModel(string name, string jsonPropertyName, double minimum, double maximum, double value, double step = 0.01)
            : base(name, jsonPropertyName, value)
        {
            this.Minimum = minimum;
            this.Maximum = maximum;
            this.Step = step;
        }
    }
}