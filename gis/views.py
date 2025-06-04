import os
from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import default_storage
from .forms import SentinelForm
from . import utils

def index(request):
    if request.method == 'POST':
        form = SentinelForm(request.POST, request.FILES)
        if form.is_valid():
            geojson_file = request.FILES['geojson_file']
            temp_path = default_storage.save('temp/' + geojson_file.name, geojson_file)
            geojson_path = os.path.join(settings.MEDIA_ROOT, temp_path)

            start_date = form.cleaned_data['start_date'].strftime("%Y-%m-%d")
            end_date = form.cleaned_data['end_date'].strftime("%Y-%m-%d")

            output_folder = os.path.join(settings.MEDIA_ROOT, 'sentinel_output')
            os.makedirs(output_folder, exist_ok=True)

            result_files = utils.export_imagery(geojson_path, start_date, end_date, output_folder)

            file_urls = {}
            for key, file_path in result_files.items():
                relative_path = os.path.relpath(file_path, settings.MEDIA_ROOT)
                file_urls[key] = settings.MEDIA_URL + relative_path.replace('\\', '/')

            return render(request, 'gis/result.html', {'file_urls': file_urls})
    else:
        form = SentinelForm()

    return render(request, 'gis/index.html', {'form': form})
