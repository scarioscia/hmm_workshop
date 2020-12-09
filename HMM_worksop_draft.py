import numpy


#=======================================#
# Create Observation and State Indicies #
#=======================================#

get_nuc_index = {
	'A' : 0,
	'C' : 1,
	'G' : 2,
	'T' : 3
}

get_state_index = {
	'Intergenic' : 0,
	'Start1' : 1,
	'Start2' : 2,
	'Start3' : 3,
	'Exon1' : 4,
	'Exon2' : 5,
	'Exon3' : 6,
	'Intron1' : 7,
	'Intron2' : 8,
	'Intron3' : 9
}


#========================#
# Initialize Empty Model #
#========================#

start_counts = numpy.zeros(10)
emission_counts = numpy.zeros((10,4))
transition_counts = numpy.zeros((10,10))


#=======================#
# Read in Training Data #
#=======================#

DNA_training_data_file = 'HMM_DNA_training.csv'
State_training_data_file = 'HMM_State_training.csv'

DNA_training_data = []
State_training_data = []

for line in open(DNA_training_data_file):
	line_as_list = line.strip().split(',')
	DNA_training_data.append(line_as_list)

for line in open(State_training_data_file):
	line_as_list = line.strip().split(',')
	State_training_data.append(line_as_list)

DNA_training_data = numpy.array(DNA_training_data)
State_training_data = numpy.array(State_training_data)


#=======================#
# Learn Training Values #
#=======================#

for row_num in range(DNA_training_data.shape[0]):
	for col_num in range(DNA_training_data.shape[1]):
		state = State_training_data[row_num,col_num]
		nucleotide = DNA_training_data[row_num,col_num]
		state_index = get_state_index[state]
		nucleotide_index = get_nuc_index[nucleotide]

		emission_counts[state_index,nucleotide_index] += 1

		if col_num < State_training_data.shape[1]-1:
			next_state = State_training_data[row_num,col_num+1]
			next_state_index = get_state_index[next_state]

			transition_counts[state_index, next_state_index] += 1

		if col_num == 0:
			start_counts[state_index] += 1


# Convert Emission Counts to Probs 
#---------------------------------

emission_probs = numpy.zeros(emission_counts.shape)

for row_num in range(emission_counts.shape[0]):
	row_sum = numpy.sum(emission_counts[row_num])
	if row_sum != 0:
		emission_probs[row_num] = emission_counts[row_num]/row_sum

# Convert Transition Counts to Probs
#-----------------------------------

transition_probs = numpy.zeros(transition_counts.shape)

for row_num in range(transition_counts.shape[0]):
	row_sum = numpy.sum(transition_counts[row_num])
	if row_sum != 0:
		transition_probs[row_num] = transition_counts[row_num]/row_sum

# Convert Start Counts to Probs
#------------------------------

start_probs = start_counts / numpy.sum(start_counts)


#===================#
# Viterbi Algorithm #
#===================#

def encode_DNA(DNA_seq):

	encoded_seq = numpy.zeros(len(DNA_seq),dtype=int)

	for i in range(len(DNA_seq)):
		nucleotide = DNA_seq[i]
		nuc_index = get_nuc_index[nucleotide]
		encoded_seq[i] = nuc_index

	return encoded_seq


def viterbi(s_probs, t_probs, e_probs, encoded_DNA_seq):

	DNA_length = encoded_DNA_seq.shape[0]
	num_states = s_probs.shape[0]

	# Initialize empty matrices
	traceback_matrix = numpy.zeros((num_states,DNA_length), dtype=int)
	traceback_matrix[:,0] = numpy.nan

	probability_matrix = numpy.zeros((num_states,DNA_length))

	# Compute the probability and traceback matrices
	for position in range(DNA_length):
		nucleotide = encoded_DNA_seq[position]
		if position == 0:
			for state in range(num_states):
				probability_matrix[state,position] = s_probs[state] * e_probs[state,nucleotide]
		else:
			for current_state in range(num_states):
				max_previous_state = None
				max_probability = None
				for previous_state in range(num_states):
					path_prob = probability_matrix[previous_state,position-1] * t_probs[previous_state, current_state] *  e_probs[current_state, nucleotide]				
					if max_probability == None or path_prob > max_probability:
						max_previous_state = previous_state
						max_probability = path_prob
				probability_matrix[current_state, position] = max_probability
				traceback_matrix[current_state, position] = max_previous_state

	# Navigate the traceback matrix
	max_path_probability = numpy.max(probability_matrix[:,-1])
	max_end_state = numpy.argmax(probability_matrix[:,-1])

	max_path = numpy.zeros(DNA_length, dtype=int)

	current_state = max_end_state
	for i in range(DNA_length-1, -1, -1): # could also use a while loop
		max_path[i] = current_state
		current_state = traceback_matrix[current_state, i]

	return max_path_probability, max_path


#=========#
# Testing #
#=========#

test_seq = 'CATGAGCTCTCGAGATCGATAGCTCTCGAGATGCGATATACGCTCGCGATGCATGCACTC'
encoded_test_seq = encode_DNA(test_seq)
viterbi_results = viterbi(start_probs, transition_probs, emission_probs, encoded_test_seq)
print(viterbi_results[1])