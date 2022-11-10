using System;
using System.Globalization;
using System.Windows.Controls;

namespace DreamHopper.UI
{
    public class IntegerValidation : ValidationRule
    {
        public override ValidationResult Validate(object value, CultureInfo cultureInfo)
        {
            string strValue = Convert.ToString(value);

            if (int.TryParse(strValue, out int _)) return new ValidationResult(true, "");
            else return new ValidationResult(false, "Value must be an integer");
        }
    }
}