import sys
from PIL import Image

from utility_functions import get_input_args_predict 
from predict_functions import load_checkpoint, predict_classes_names, predict

def main():
	# Calling the get_input_args_predict function to get the input arguments.
	in_arg = get_input_args_predict()

	# checking what device the user want to train by.
	if in_arg.gpu:
	    device = 'cuda'
	else:
	    device = 'cpu'


	# Using try statement to check if the passed image_path by the user is available or not.
	image_path = in_arg.input
	try:
	    Image.open(image_path)
	except:
		print("The Image path is not Defined!")
		print("Please try again with the correct path of the Image!")
		sys.exit(0)


	cat_to_name = in_arg.category_names
	# The dfault of cat_to_name is None, so if it isn't None and the user passed a value, check the file.
	if cat_to_name != None:
		# Using try statement to check if the passed category_names by the user is available or not.
		try:
			with open(cat_to_name, 'r'):
				pass
		except FileNotFoundError:
			print("The path of Category Names file is not Defined!")
			print("Please try again with the correct path!")
			sys.exit(0)


	filepath = in_arg.checkpoint
	# Loading the model by passing the path of it, if it's available 
	# and the device which the user want to predict with.
	model = load_checkpoint(filepath, device)
	

	topk = in_arg.top_k
	# Calling predict_classes_names to obtain the most likely 5 predicted classes with their prbabilities. 
	probs, classes = predict(image_path, model, device, topk)

	# If there's a categories to names file is passed, call predict_classes_names function to 
	# create the category names from the classes labels.
	# Then adjust the word variable for the result printing process.
	if cat_to_name != None:
		classes = predict_classes_names(cat_to_name, classes_output = classes)
		word = "Flower Name"

	else:
		word = "Class Number"



	if len(probs) == 1:
		# if topk = 1, the probs list will contain a single value, so print once.
	    print("Predicted {}: {}\t\t ... \t Predicted Class Probability: {:.3f}".format(word, classes[0], probs[0]))

	else:
		# if topk > 1, the probs list will contain multiple value, so loop through them and to print.
	    print("Top {} most likely Classes".format(topk))
	    for i in range(len(probs)):
	        print("Predicted {} ({}): {:>20}\t ... \tPredicted Class Probability: {:.3f}".format(word, i+1, classes[i], probs[i]))


if __name__ == "__main__":
    main()