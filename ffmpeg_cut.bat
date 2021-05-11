set arg1=%1
set arg2=%2
set arg3=%3


C:\ffmpeg\bin\ffmpeg.exe -i %arg1% -vcodec copy -acodec copy -ss %arg2% -t %arg3% %arg1%_out.avi