rd /s /q ..\static
mkdir ..\static
robocopy .\build\static ..\static /e
robocopy .\build\ ..\static /xf *.html
robocopy .\build\ ..\templates *.html