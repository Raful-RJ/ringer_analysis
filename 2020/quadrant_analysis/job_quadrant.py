import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import matplotlib.image as mpimg
from matplotlib.ticker import PercentFormatter
from PyPDF2 import PdfFileMerger
from PIL import Image
import glob 

import argparse

##################################################################################
##################################################################################
### This script aims to perform quadrant analizes over certain shower shapes   ###
### considering two different selection chains.				       ###	   					       ###
##################################################################################
##################################################################################

parser = argparse.ArgumentParser(description = 'This script aims to perform quadrant analizes over certain shower shapes considering two different selection chains. In  this case, MC and Offline chains will be crutinized in order to identify agreements and disagreements btw these two chains.', add_help = False)
#parser = argparse.ArgumentParser()

parser.add_argument('-i','--inputFiles', action='store', 
    dest='inputFiles', required = True, nargs='+',
    help = "The input files that will be used to generate the plots (.npz)")

parser.add_argument('-o','--outputFile', action='store', 
    dest='outputpath', required = False, default = None,
    help = "The output store name where the plots will be saved in.")

parser.add_argument('-v','--caloVar', action='store', 
    dest='var_list', required = False, nargs='+', default = ['rhad','reta','eratio','weta2'],
    help = "The set of calorimeter variables to be analyzed in this task")

parser.add_argument('-p','--pidname', action='store', 
    dest='pidname', required = False, nargs = '+', default = ['MC_Truth','ph_tight'],
    help = "Selection methods to be analyzed")

parser.add_argument('-m','--modelpath', action='store', 
    dest='modelpath', required = False, nargs = '+', default = [],
    help = "Path where NN selection method is located (must be ordered)")

parser.add_argument('-a','--analyse', action='store', type = int, 
    dest='analyse', required = False, default = 0,
    help = "Specifing events: \n0 - All\n1 - Only Sgn\n2 - Only Bkg")


import sys,os
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args()

var_list = args.var_list
output_path = args.outputpath
data_file_list = args.inputFiles
if output_path == None: output_path = os.getcwd()
#output_path = os.getcwd() if output_path == None else None
pidname = args.pidname # We will only be looking at this operation point
model_path = args.modelpath
analyse = args.analyse

print('*'*15)
print('Shower Shape List: ', var_list)
print('Pidname: ', pidname)
print('Output path: ', output_path)
print('Data Set Path Files: ', data_file_list)
print('Model Path Files: ', model_path)
print('Analyse: ', analyse)
print('*'*15)

class Quadrant():

	def __init__(self, pidname, data_file_list, var_list, output_path, model_path, analyse):


		self.pidname = pidname
		self.analyse = analyse
		self.var_dict = dict.fromkeys(var_list,None)
		self.output_path = output_path
		self.model_path = model_path
		self._var_dict_latex = {'reta': '$R_{\eta}$',
				  'rhad': '$R_{had}$',
				  'eratio': '$E_{ratio}$',
				  'weta2': '$\omega_{\eta 2}$'}
		self._etBins = [15,20,30,40,50,100000]
		self._etaBins = [0.0,0.80,1.37,1.54,2.37, 2.54]
		self._known_models = {'MC_Truth': 'MC',
				      'ph_tight': 'Offline Tight',
				      'ph_medium': 'Offline Medium' ,
				      'ph_loose' :'Offline Loose',
				      'T0HLTPhotonT2CaloLoose': 'T2Calo Loose',
				      'T0HLTPhotonT2CaloMedium': 'T2Calo Medium',
				      'T0HLTPhotonT2CaloTight': 'T2Calo Tight'}
				

		for data_file in data_file_list:
			data_set = np.load(data_file)		
			et = data_set['etBinIdx']
			eta = data_set['etaBinIdx']
			model_exists = True
			for path in self.model_path:
				path_json = glob.glob(path + '/models/*et%i_eta%i.json' %(et,eta))
				if len(path_json) == 0: model_exists = False
			if model_exists:
				self.compile(data_set)			
	
	def compile(self, data_set):

		self._list_of_truths = [] #[[m1_accept, m1_reject],[m2_accept,m2_reject]]
		for var in self.var_dict.keys():
			self.var_dict[var] = np.where(data_set['features'] == var)[0][0]

		print('Computando eventos aprovados ou rejeitados por %s e %s na regiao et%i_eta%i...' %(self.pidname[0], self.pidname[1],data_set['etBinIdx'],data_set['etaBinIdx']))
		aux = 0		
		Idx_mctype = np.where(data_set['features'] == 'mc_type')[0][0]
		if self.analyse == 1:			
			e_data_set = data_set['data'][self.get_MC_Truth(Idx_mctype,data_set['data'])[0]]
		elif self.analyse == 2:
			e_data_set = data_set['data'][self.get_MC_Truth(Idx_mctype,data_set['data'])[1]]
		else: e_data_set = data_set['data']

		for idx, pidname in enumerate(self.pidname):
			
			if pidname in self._known_models.keys():
				aux += 1	
				if pidname == 'MC_Truth':
					print('Computando verdades de MC')
					self._list_of_truths.append(self.get_MC_Truth(Idx_mctype,e_data_set))
				else:
					print('Computando verdades do offline')
					self._list_of_truths.append(self.get_Offline_Truth(pidname,e_data_set, data_set))
			else:
				self._known_models[pidname.upper()] = pidname
				print('Carregando modelo %s em et%i_eta%i' %(pidname,data_set['etBinIdx'],data_set['etaBinIdx']))
				self._list_of_truths.append(self.get_NN_model_Truths(idx-aux, e_data_set, data_set))

		# Get index of relevant collumns
		print('Computando intersecoes de eventos aprovados ou rejeitados por %s e %s na regiao et%i_eta%i...' %(self.pidname[0], self.pidname[1],data_set['etBinIdx'],data_set['etaBinIdx']))
		# Where both agrees
		Idx_both_agrees = np.intersect1d(self._list_of_truths[0][0],self._list_of_truths[1][0])
		# Where MC accepts and Offline rejects
		Idx_m1_not_m2 = np.intersect1d(self._list_of_truths[0][0], self._list_of_truths[1][1])
		# Where MC rejects and Offline accepts
		Idx_not_m1_m2 = np.intersect1d(self._list_of_truths[0][1], self._list_of_truths[1][0])
		# Where both disagrees
		Idx_both_disagrees = np.intersect1d(self._list_of_truths[0][1], self._list_of_truths[1][1])

		for var in self.var_dict.items():
			var_both_agrees = data_set['data'][Idx_both_agrees][:,var[1]]
			var_m1_not_m2 = data_set['data'][Idx_m1_not_m2][:,var[1]]
			var_not_m1_m2 = data_set['data'][Idx_not_m1_m2][:,var[1]]
			var_both_disagrees = data_set['data'][Idx_both_disagrees][:,var[1]]
			print('Plotando quadrante da variavel %s na regiao et%i_eta%i...' %(var[0],data_set['etBinIdx'],data_set['etaBinIdx']))
			self.plot(var[0], 150 , var_both_agrees,var_m1_not_m2,var_not_m1_m2, var_both_disagrees, et_bin = data_set['etBinIdx'], eta_bin = data_set['etaBinIdx'])

	def plot(self, var, n_bins = 200, *args, **kwargs ):

		def hist_range(data, k):
			avg = np.mean(data)
			std = np.std(data, ddof = 1)
			range = (avg - k*std, avg + k*std)
			var_min = np.min(data)
			var_max = np.max(data)
			return [var_min, var_max, range[0], range[1], (range[1] - range[0])]

		et_bin = kwargs['et_bin']
		eta_bin = kwargs['eta_bin']

		output_path = self.output_path + '/quadrant_et%i_eta%i_%s_%s_Vs_%s.png' %(et_bin,eta_bin,var,self.pidname[0],self.pidname[1])
		if var == 'rhad':
			#array_ranges = np.array([hist_range(arg,2) for arg in args])
			#arange = (np.mean(array_ranges[:,2]),np.mean(array_ranges[:,3]))
			arange = (-0.3,0.3)
		elif var == 'weta2':
			arange = (0,0.02)
		else: #eratio e reta
			arange = (0.75,1.1)

		print('Construindo bins e agrupando os eventos ...')
		bins = np.linspace(arange[0],arange[1],n_bins + 1)

		list_occurences = []
		list_rel_occurences = []

		N_agreements = len(args[0]) + len(args[3])
		N_disagreements = len(args[1]) + len(args[2])
		# 1st col | 2nd col | 3rd col | 4th col | 5th col
		for idx, lim_inf in enumerate(bins[:-1]):
			n_both_accept = np.where((args[0]>=lim_inf)*(args[0]<=bins[idx+1]))[0].shape[0]
			n_1_accept_2_reject = np.where((args[1]>=lim_inf)*(args[1]<=bins[idx+1]))[0].shape[0]
			n_2_accept_1_reject = np.where((args[2]>=lim_inf)*(args[2]<=bins[idx+1]))[0].shape[0]
			n_both_reject = np.where((args[3]>=lim_inf)*(args[3]<=bins[idx+1]))[0].shape[0]
			center = 0.5*(lim_inf + bins[idx+1])
			n_events = (n_both_accept + n_both_reject + n_1_accept_2_reject + n_2_accept_1_reject)/100

			if n_events != 0:
				list_occurences.append([center, n_both_accept, n_1_accept_2_reject, n_2_accept_1_reject, n_both_reject])
				list_rel_occurences.append([center, n_both_accept/n_events, n_1_accept_2_reject/n_events, n_2_accept_1_reject/n_events, n_both_reject/n_events])
			else:
				list_occurences.append([center, 0, 0, 0, 0])
				list_rel_occurences.append([center, 0, 0, 0, 0])

		array_occurences = np.array(list_occurences)
		array_rel_occurences = np.array(list_rel_occurences)

		
		print('Configurando parametros da plot ...')
		#fig,ax = plt.subplots(nrows=2,ncols=1, sharex=True, sharey=False, height_ratios = [2,1])
		fig = plt.figure(dpi = 200, figsize=(8, 6))
		G = grd.GridSpec(2,1, height_ratios = [2,1])
		fig_ax1 = fig.add_subplot(G[0,0])
		fig_ax2 = fig.add_subplot(G[1,0])
		pidname_legend = []
		for pid in self.pidname:
			try:				
				pidname_legend.append(self._known_models[pid])
			except:
				pidname_legend.append(self._known_models[pid.upper()])
		legend = ['Both Approved','%s Approved &\n %s Rejected' %(pidname_legend[0],pidname_legend[1]), '%s Rejected &\n %s Approved' %(pidname_legend[0],pidname_legend[1]),'Both Rejected']
		color = ['k', 'red', 'blue', 'gray' ]
		#idx_max = np.where(array_ranges[:,-1] == np.max(array_ranges[:,-1]))[0][0]
		#range = (array_ranges[idx_max][0],array_ranges[idx_max][1])

		list_ymax = []
		for axis in range(4):
			#fig_ax1.plot(array_occurences[:,0], array_occurences[:,axis+1], '.', color = color[axis], alpha = 0.75, label = legend[axis])
			fig_ax1.step(bins[1:], array_occurences[:,axis+1], color = color[axis],linewidth = 0.7, alpha = 0.7, label = legend[axis])
			list_ymax.append(np.max(array_occurences[:,axis+1]))

		ymax = np.max(list_ymax)
		xmin = np.min(bins[1:])
		total = N_agreements+N_disagreements
		fig_ax1.text(xmin,ymax,'Total of events: %i\nAgreements: %.2f%%\nDisagreements: %.2f%%' %(total,100*N_agreements/total,100*N_disagreements/total),ha='left', va='top',fontsize = 'small')
		#fig_ax1.hist([args[3],args[1],args[2],args[0]],n_bins,range = range, alpha = 0.75, label= legend, histtype = 'step', log = True, color = ['gray','red','blue','k'])
		#n_events = args[0].shape[0] + args[1].shape[0] + args[2].shape[0] + args[3].shape[0]
		#w1 = 100*np.ones(args[0].shape[0])/(n_events)
		#w2 = 100*np.ones(args[3].shape[0])/(n_events)
		#fig_ax2.hist([args[0],args[3]],n_bins, range = range, weights= (w1,w2), alpha = 0.75, histtype = 'step', color = ['k','gray'])
		#fig_ax2.hist(args[3],200, density = True, alpha = 0.7, histtype = 'step', log = True, color = 'gray')
		fig_ax2.step(bins[1:], array_rel_occurences[:,1], linewidth = 0.7, color = 'k', alpha = 0.7)
		fig_ax2.step(bins[1:], array_rel_occurences[:,4], linewidth = 0.7, color = 'gray', alpha = 0.7)

		fig_ax3 = fig_ax2.twinx()
		#w1 = 100*np.ones(args[1].shape[0])/(n_events)
		#w2 = 100*np.ones(args[2].shape[0])/(n_events)
		#fig_ax3.hist([args[1],args[2]],n_bins, range = range, weights= (w1,w2), alpha = 0.75, histtype = 'step', color = ['red','blue'])
		#fig_ax3.hist(args[2],200, density = True, alpha = 0.7, histtype = 'step', log = True, color = 'blue')
		fig_ax3.step(bins[1:], array_rel_occurences[:,2], linewidth = 0.7, color = 'red', alpha = 0.7)
		fig_ax3.step(bins[1:], array_rel_occurences[:,3], linewidth = 0.7, color = 'blue', alpha = 0.7)

		#fig_ax1.set_yticks(np.arange(10,np.max(array_occurences[:,4])*100,10))
		fig_ax1.set_yticks([1e2,1e3,1e4,1e5,1e6,1e7])
		fig_ax1.xaxis.set_ticklabels([])
		fig_ax2.set_xlabel(self._var_dict_latex[var], fontsize = 'x-large', loc = 'right' )
		fig_ax1.set_yscale('log')
		fig_ax1.set_ylabel("Count")
		fig_ax2.set_ylabel("Agreement [%]", fontsize = 'small')
		fig_ax3.set_ylabel('Disagreement [%]', fontsize = 'small')
		fig_ax1.legend(loc = "upper right", edgecolor = 'white', framealpha = 0.5, fontsize = 'xx-small')
		fig_ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100))
		fig_ax3.yaxis.set_major_formatter(PercentFormatter(xmax=100))
		

		fig_ax1.set_title('%i $ < E_{T} < $ %i [GeV]' %(self._etBins[et_bin],self._etBins[et_bin+1]) + " "*7 + '%.2f $ < \eta < $ %.2f' %(self._etaBins[eta_bin],self._etaBins[eta_bin+1]),fontsize = 'large')

		plt.savefig(output_path)
		print('Plot de quadrante %s Vs. %s da variavel %s na regiao et%i_eta%i salvo em : %s' %(self.pidname[0],self.pidname[1], var, et_bin,eta_bin,output_path))

	def get_MC_Truth(self,Idx_mctype, e_data_set):
			""" (int,array) --> (1st_array, 2nd_array)
			1st array: where mc accepts
			2nd aray: where mc rejects """
			mc_accept = np.where( (e_data_set[:,Idx_mctype] == 13) + (e_data_set[:,Idx_mctype] == 14) + (e_data_set[:,Idx_mctype] == 15) )
			mc_reject = np.where( (e_data_set[:,Idx_mctype] != 13) * (e_data_set[:,Idx_mctype] != 14) * (e_data_set[:,Idx_mctype] != 15) )
			
			return [mc_accept, mc_reject]

	def get_Offline_Truth(self, pidname,e_data_set, data_set):
		Idx_pidname = np.where(data_set['features'] == pidname)[0][0]
		return [np.where(e_data_set[:,Idx_pidname] == 1), np.where(e_data_set[:,Idx_pidname] != 1)]

	def get_NN_model_Truths(self, idx,e_data_set, data_set):

		from tensorflow.keras.models import Sequential
		from tensorflow.keras.layers import Dense
		from tensorflow.keras.models import model_from_json		

		def open_conf_file(path):
			with open(path,'r') as File:
				lines = File.readlines()

			lines = [_.replace('\n','') for _ in lines] #removing \n
			lines = [_.replace(' ','') for _ in lines] #removing spaces
			lines = [_.replace(';',',') for _ in lines] #removing ;

			lista = [_.split(':') for _ in lines]
			dict_conf = ([ [_[0],_[1].split(',',-1)] for _ in lista])

			for idx, item in enumerate(dict_conf):
				for idx_2, _ in enumerate(item[1]):
					try: dict_conf[idx][1][idx_2] = float(_)
					except ValueError: None
			return dict(dict_conf)

		def norma_1(matrix):
			n_1 = np.linalg.norm(matrix, ord = 1, axis = 1)
			for i,linha in enumerate(matrix):
				matrix[i] = linha/n_1[i]
			return matrix

		et, eta = data_set['etBinIdx'], data_set['etaBinIdx']
		path_json = glob.glob(self.model_path[idx] + '/models/*edium*et%i_eta%i.json' %(et,eta))[0]
		path_h5 = glob.glob(self.model_path[idx] + '/models/*edium*et%i_eta%i.h5' %(et,eta))[0]
		path_conf = glob.glob(self.model_path[idx] + '/*edium*.conf')[0]
		dict_conf = open_conf_file(path_conf)

		print('Construindo modelo e carregando pesos ...')
		with open(path_json, 'r') as json_file:
			model = model_from_json(json_file.read())
		model.load_weights(path_h5,'r')

		print('Propagando data set ...')
		predictions = np.transpose(model.predict(norma_1(e_data_set[:,1:101])))[0]

		print('Carregando Thresholds ...')
		if len(dict_conf['Threshold__slope']) == 20:
			slope = np.reshape(np.array(np.array(dict_conf['Threshold__slope'])),(5,4))[et][eta]
			offset = np.reshape(np.array(np.array(dict_conf['Threshold__offset'])),(5,4))[et][eta]
		else:
			slope = np.reshape(np.array(np.array(dict_conf['Threshold__slope'])),(5,5))[et][eta]
			offset = np.reshape(np.array(np.array(dict_conf['Threshold__offset'])),(5,5))[et][eta]
		
		threshold = slope*e_data_set[:,0] + offset

		return [np.where(predictions >= threshold), np.where(predictions < threshold)]



def convert2pdf(pidname, output_path):

	print('Iniciando conversao e merge das plots em pdf ...')

	def create_dict(list_img_path):

		dict_img = {}

		for img_path in list_img_path:

			img_info = img_path.split('/')[-1].split('_')[1:]
			et_bin = int(img_info[0][-1])
			eta_bin = int(img_info[1][-1])
			phase_space = 'et%i_eta%i' %(et_bin,eta_bin)
		

			if phase_space not in dict_img.keys():
				dict_img[phase_space] = [img_path] 
			else:
				dict_img[phase_space].append(img_path)

		return dict_img

	all_image_paths = glob.glob(output_path + '/*.png')
	dict_img= create_dict(all_image_paths)

	print('Construindo arquivos temporarios ...')
	for keys, set_imgs in dict_img.items():
		cp_set_imgs = set_imgs[:]
		n_figs = len(set_imgs)
		n_cols_rows = round(n_figs**.5)

		fig, ax = plt.subplots(nrows=n_cols_rows,ncols=n_cols_rows, dpi=400)

		for row in range(n_cols_rows):
			for col in range(n_cols_rows):				
				if cp_set_imgs != []:
					img = cp_set_imgs.pop()
					ax[row,col].imshow(mpimg.imread(img))
					ax[row,col].get_xaxis().set_visible(False)
					ax[row,col].get_yaxis().set_visible(False)
					[s.set_visible(False) for s in ax[row,col].spines.values()]
					ax[row,col].tick_params(left=False, labelleft=False) #remove ticks
					 #remove box
				else: break
		fig.tight_layout()
		plt.savefig(output_path + '/allvar_' + keys + '_%s_Vs_%s' %(pidname[0],pidname[1]) + '.pdf')

	merger = PdfFileMerger()

	print('Concatenando arquivos ...')
	for pdf in glob.glob(output_path + '/allvar*'):
		merger.append(pdf)
	merger.write(output_path + '/quadrant' + '_%s_Vs_%s' %(pidname[0],pidname[1]) + '.pdf')
	merger.close()

	print('Eliminando arquivos temporarios ...')
	for del_img in glob.glob(output_path + '/allvar*'):
		os.system('rm %s' %(del_img))
	print('Task finalizada!')		

Quadrant(pidname, data_file_list, var_list, output_path, model_path, analyse)
convert2pdf(pidname,output_path)

		
		
