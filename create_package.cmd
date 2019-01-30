echo OFF
echo[
echo[
echo                   ###########################
echo                   ##   Building BlasNet    ##
echo                   ###########################
echo[
echo[

dotnet build -c Release

echo BlasNet was built in release configuration, ready to create package
pause
echo[
echo[
echo                 ##################################
echo                 ##   Building BlasNet Package   ##
echo                 ##################################
echo[
echo[

nuget pack Proxem.BlasNet/nuspec/Proxem.BlasNet.nuspec -Symbols -OutputDirectory Proxem.BlasNet/nuspec/

echo Package was created in folder Proxem.BlasNet/nuspec/
pause