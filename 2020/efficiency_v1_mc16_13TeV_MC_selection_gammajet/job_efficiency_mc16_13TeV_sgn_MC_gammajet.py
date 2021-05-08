

from prometheus import EventATLAS
from prometheus.enumerations import Dataframe as DataframeEnum
from Gaugi.messenger import LoggingLevel, Logger
from Gaugi import ToolSvc, ToolMgr
import argparse
mainLogger = Logger.getModuleLogger("job")
parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()


parser.add_argument('-i','--inputFiles', action='store', 
    dest='inputFiles', required = True, nargs='+',
    help = "The input files that will be used to generate the plots")

parser.add_argument('-o','--outputFile', action='store', 
    dest='outputFile', required = False, default = None,
    help = "The output store name.")

parser.add_argument('-n','--nov', action='store', 
    dest='nov', required = False, default = -1, type=int,
    help = "Number of events.")

import sys,os
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args()



acc = EventATLAS( "EventATLASLoop",
                  inputFiles = args.inputFiles, 
                  treePath= '*/HLT/PhysVal/Egamma/photons',
                  dataframe = DataframeEnum.Photon_v1, 
                  outputFile = args.outputFile,
                  level = LoggingLevel.INFO
                  )



from EventSelectionTool import EventSelection, SelectionType, EtCutType

evt = EventSelection('EventSelection')
#evt.setCutValue( SelectionType.SelectionOnlineWithRings )
#pidname = 'MediumLLH_DataDriven_Rel21_Run2_2018'
evt.setCutValue( EtCutType.L2CaloAbove , 15)
evt.setCutValue( SelectionType.SelectionPhoton )

ToolSvc += evt


from TrigEgammaEmulationTool import Chain, Group, TDT

triggerList = [
                Group( TDT( "TDT_g10_etcut", "HLT_g10_etcut"), None, 20 ),
                Group( Chain( "EMU_g20_tight_noringer_L1EM3","L1_EM3","HLT_g20_tight_noringer"), None, 20 ),
                Group( Chain( "EMU_g20_medium_noringer_L1EM3","L1_EM3","HLT_g20_medium_noringer"), None, 20 ),
                Group( Chain( "EMU_g20_loose_noringer_L1EM3","L1_EM3","HLT_g20_loose_noringer"), None, 20 ),
                Group( Chain( "EMU_g20_tight_ringer_v1_L1EM3","L1_EM3","HLT_g20_tight_ringer_v1"), None, 20 ),
                Group( Chain( "EMU_g20_medium_ringer_v1_L1EM3","L1_EM3","HLT_g20_medium_ringer_v1"), None, 20 ),
                Group( Chain( "EMU_g20_loose_ringer_v1_L1EM3","L1_EM3","HLT_g20_loose_ringer_v1"), None, 20 ),
                



                ]
                




from EfficiencyTools import EfficiencyTool
alg = EfficiencyTool( "Efficiency" )


for group in triggerList:
  alg.addGroup( group )

ToolSvc += alg

acc.run(args.nov)
