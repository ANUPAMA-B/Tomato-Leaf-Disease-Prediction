import zipfile
import io
zf = zipfile.ZipFile("tomato.zip", "r")
zf.extractall("/content")
zf.close()