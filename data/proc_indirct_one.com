#!/bin/csh -f

if ($#argv < 2) then
    echo "Usage: proc_indirct_one.com input.ft1 output.ft"
    exit 1
endif

set inFile = "$1"
set outFile = "$2"
set tmpDir = "./nmrpipe_tmp_$$"

mkdir -p ${tmpDir}/ft2
mkdir -p ${tmpDir}/ft

xyz2pipe -in ${inFile} -x -verb \
| nmrPipe -fn TP \
| nmrPipe -fn SP -off 0.50 -end 0.95 -pow 1 -elb 0.0 -glb 0.0 -c 0.5 \
| nmrPipe -fn ZF -zf 1 -auto \
| nmrPipe -fn FT \
| nmrPipe -fn PS -p0 0.0 -p1 0.0 -di \
| nmrPipe -fn TP \
| nmrPipe -fn POLY -auto \
| pipe2xyz -out ${tmpDir}/ft2/test%05d.ft2 -x -ov

xyz2pipe -in ${tmpDir}/ft2/test%05d.ft2 -z -verb \
| nmrPipe -fn SP -off 0.50 -end 0.95 -pow 1 -elb 0.0 -glb 0.0 -c 0.5 \
| nmrPipe -fn ZF -zf 1 -auto \
| nmrPipe -fn FT \
| nmrPipe -fn PS -p0 0.0 -p1 0.0 -di \
| pipe2xyz -out ${tmpDir}/ft/test%03d.ft3 -z -ov

xyz2pipe -in ${tmpDir}/ft/test%03d.ft3 -z -verb \
  > ${outFile}

/bin/rm -rf ${tmpDir}
