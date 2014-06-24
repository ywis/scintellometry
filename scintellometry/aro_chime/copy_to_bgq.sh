#/bin/bash
FILES=( $(ls ${1} -v) )
NUMFILES=`ls -l ${1}| wc -l`

cd ${1}
ssh klages@bgqdev.scinet.utoronto.ca "mkdir /scratch/p/pen/klages/ARO/${1}"
mkdir ~/python_local_temp/${1}

for (( i=0; i<=(${NUMFILES}/1000); i++ ))
do
	start=$((i*1000))
	outfilename=`printf '%07d.dat' ${i}`
	echo ${start} 1>&2
	echo ${FILES[@]:${start}:1000} 1>&2
	echo "${outfilename}.dat" 1>&2

    #remote python execution is failing... make the big file locally
	cat ${FILES[@]:${start}:1000} | python ~/decode_stream.py ~/python_local_temp/${1}/${outfilename}

	#copy it over
    scp -oCipher=NONE ~/python_local_temp/${1}/${outfilename} klages@bgqdev.scinet.utoronto.ca:/scratch/p/pen/klages/ARO/${1}/${outfilename}
    scp -oCipher=NONE ~/python_local_temp/${1}/${outfilename}.meta klages@bgqdev.scinet.utoronto.ca:/scratch/p/pen/klages/ARO/${1}/${outfilename}.meta

    #clean up locally
	rm ~/python_local_temp/${1}/${outfilename}
	rm ~/python_local_temp/${1}/${outfilename}.meta

	#check for errors with the rm command (generally caused by getting to only symlinks)
	if [ $? -ne 0 ]
	then
		break
	fi
done

if [ $? -ne 0 ]
then
	echo Exited ${1} early at ${i} so please check all real files were transferred | mail -s "done processing ${1} with possible error" pklages@gmail.com
else
	echo ${1} done processing | mail -s "done processing ${1}" pklages@gmail.com
fi

