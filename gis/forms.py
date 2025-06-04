from django import forms
from datetime import date

class SentinelForm(forms.Form):
    geojson_file = forms.FileField(label="GeoJSON File")
    start_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    end_date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))

    def clean(self):
        cleaned_data = super().clean()
        start = cleaned_data.get('start_date')
        end = cleaned_data.get('end_date')
        if start and end:
            if end < start:
                raise forms.ValidationError("End date must be after start date.")
            if start > date.today():
                raise forms.ValidationError("Start date cannot be in the future.")
        return cleaned_data
