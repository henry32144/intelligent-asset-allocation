rd /s /q ..\static
mkdir ..\static

robocopy .\landing\build\static ..\static /e
robocopy .\signup\build\static ..\static /e
robocopy .\landing\build\ ..\static /xf *.html
robocopy .\landing\build\ ..\templates *.html

rename .\dashboard\build\index.html dashboard.html
robocopy .\dashboard\build\static ..\static /e
robocopy .\dashboard\build\ ..\static /xf *.html
robocopy .\dashboard\build\ ..\templates *.html

rename .\login\build\index.html login.html
robocopy .\login\build\static ..\static /e
robocopy .\login\build\ ..\static /xf *.html
robocopy .\login\build\ ..\templates *.html

rename .\signup\build\index.html signup.html
robocopy .\signup\build\static ..\static /e
robocopy .\signup\build\ ..\static /xf *.html
robocopy .\signup\build\ ..\templates *.html