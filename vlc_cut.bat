
set arg1=%1
set arg2=%2
set arg3=%3
shift
shift

"C:\Program Files\VideoLAN\VLC\vlc.exe" -vvv %arg1% --start-time=%arg2% --stop-time=%arg3% --sout file/ps:%arg1%.mpg
