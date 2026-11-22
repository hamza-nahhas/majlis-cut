; Windows installer for majlis-cut, built by tools\build_installer.ps1:
;   ISCC /DMyVersion=0.4.1 /DSourceDir=build\app /Obuild installer\majlis-cut.iss
; SourceDir holds the tool's files plus python\ (Python with the libraries already
; installed) and ffmpeg\bin\. The tool is installed for the current user only, so
; no administrator rights are needed and the in-app updater can write its files.

#ifndef MyVersion
  #error Pass /DMyVersion=X.Y.Z
#endif
#ifndef SourceDir
  #error Pass /DSourceDir=<folder with the tool, python and ffmpeg>
#endif

[Setup]
AppId={{6B0F2E61-5C2A-4E7B-9D1A-3F4C8E2B7A90}
AppName=majlis-cut
AppVersion={#MyVersion}
AppVerName=majlis-cut {#MyVersion}
AppPublisher=hamza-nahhas
AppPublisherURL=https://github.com/hamza-nahhas/majlis-cut
AppSupportURL=https://github.com/hamza-nahhas/majlis-cut/issues
AppUpdatesURL=https://github.com/hamza-nahhas/majlis-cut/releases
DefaultDirName={localappdata}\Programs\majlis-cut
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputBaseFilename=majlis-cut-setup-v{#MyVersion}
SetupIconFile=..\majlis_cut\icon.ico
UninstallDisplayIcon={app}\majlis_cut\icon.ico
UninstallDisplayName=majlis-cut
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes

[Languages]
Name: "arabic"; MessagesFile: "compiler:Languages\Arabic.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
arabic.ShortcutName=مونتاج الدروس
english.ShortcutName=majlis-cut
arabic.OpenNow=افتح الأداة الآن
english.OpenNow=Open majlis-cut now

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[InstallDelete]
; a newer installer brings its own Python and libraries: never mix them with the old ones
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\ffmpeg"
Type: filesandordirs; Name: "{app}\majlis_cut"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\{cm:ShortcutName}"; Filename: "{app}\start.bat"; WorkingDir: "{app}"; IconFilename: "{app}\majlis_cut\icon.ico"
Name: "{autodesktop}\{cm:ShortcutName}"; Filename: "{app}\start.bat"; WorkingDir: "{app}"; IconFilename: "{app}\majlis_cut\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\start.bat"; WorkingDir: "{app}"; Description: "{cm:OpenNow}"; Flags: postinstall nowait skipifsilent shellexec

[UninstallDelete]
; what the tool made inside its own folder (the lessons themselves are elsewhere)
Type: filesandordirs; Name: "{app}"
