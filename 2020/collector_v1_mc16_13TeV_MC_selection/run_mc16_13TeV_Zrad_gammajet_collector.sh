BASEPATH_gammajet=/home/juan.marin/datasets/physval/user.jlieberm.mc16_13TeV.4231*/*.root

prun_jobs.py -c "python3 job_collector.py --Zrad" -mt 40 -i $BASEPATH_gammajet

mkdir gammajetMC
rm *.root
mv output* gammajetMC
