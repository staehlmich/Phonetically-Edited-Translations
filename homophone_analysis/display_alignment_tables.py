## The program creates an HTML table for each alignment line 
## in order to visualize the alignment matrix as computed by GIZA.
##
## The format of the alignment lines in the input file (example lines) is like this:
#### 0-0 0-1 1-2 2-3 3-4 3-5 4-6 5-7
#### 0-0
#### 0-0 1-1 2-2 3-3 7-4 9-5 4-6 5-6 6-6 7-6 8-6 9-6 9-7 9-8 10-9 
##
## New: The words in the top row are now written vertically in order to generate
##      equal space for each column (thanks to Mathias Mueller).
## New: It is now possible to specify the start line number and the number of tables
##      to be created --> number_of_tables
##
## by Martin Volk adapted by Michael Staehli

import argparse
import sys  ## module for command line arguments

def display_alignment_tables(lang1_filename, lang2_filename, alignment_filename):
	#TODO: optional arguments
	if len(sys.argv) > 4:
		## the start line number in the alignment file
		start_line_num = int(sys.argv[4])
	else:
		start_line_num = 0

	if len(sys.argv) > 5:
		## the number of tables to be created
		number_of_tables = int(sys.argv[5])
	else:
		## the number of alignment tables which shall be created by the program
		number_of_tables = 10

	#################################################################

	## open the alignment file for reading
	alignment_infile = open(alignment_filename, 'r')

	## open the language_1 file for reading
	lang1_infile = open(lang1_filename, 'r')

	## open the language_2 file for reading
	lang2_infile = open(lang2_filename, 'r')

	## counter for the number of alignment lines
	line_count = 0

	## skip the lines before the start line
	while line_count < start_line_num:
		line_count += 1
		junk_line = alignment_infile.readline()
	#	print 'Test ', junk_line
		lang1_infile.readline()
		lang2_infile.readline()

	print('<html>')
	print('<head>')
	print('  <meta http-equiv="Content-Type" content="text/html;charset=UTF-8">')
	print('  <title>Word Alignments</title>')
	## the attribute 'border-collapse' ensures narrow lines between the table cells
	print('  <style type = "text/css">')
	print('    h1 { font-family: arial}')
	print('    h3 { font-family: arial}')
	print('    td { font-family: arial}')
	print('    table { table-layout:fixed;')
	print('            border-collapse: collapse; }')
	print('    table, th, td { border: 1px solid black;')
	print('                    table-layout:fixed; }')
	print('    thead td { vertical-align:bottom;')
	print('               height:80px; }')
	print('    td .vertical {')
	print('      -ms-transform:rotate(-90deg); /* IE 9 */')
	print('      -moz-transform:rotate(-90deg); /* Firefox */')
	print('      -webkit-transform:rotate(-90deg); /* Safari and Chrome */')
	print('      -o-transform:rotate(-90deg); /* Opera */')
	print('      transform-origin: center center 0;')
	print('      width: 1em;')
	print('      white-space:nowrap; }')
	print('  </style>')
	print('</head>')
	print('</body>')
	print('  <h2>Word Alignments</h2>')
	print('  <p>Input files:</p>')
	print('  <ol>')
	print('    <li>', lang1_filename, '</li>')
	print('    <li>', lang2_filename, '</li>')
	print('    <li>', alignment_filename, '</li>')
	print('  </ol>')

	## for each line in the alignment file
	for line in alignment_infile:
		line_count += 1
		# strip away the \n symbol
		line = line.rstrip()
		# split the line into a list of strings
		line_list = line.split()

		## initialize the variables
		max_x = 0
		max_y = 0
		line_tuples = []

		## compute the size of the needed matrix
		## ... and save the alignment tuples in a list of tuples
		for pair in line_list:
			(string_x, string_y) = pair.split('-')
			# convert the strings to numbers
			x = int(string_x)
			y = int(string_y)
			if x > max_x:
				max_x = x
			if y > max_y:
				max_y = y
			# save the current x-y tuple in a list of tuples
			line_tuples.append((x, y))

		## initialize and fill the matrix
		alignment_matrix = []
		new = []
		for x in range (0, max_x+1):
			for y in range (0, max_y+1):
				## if this alignment is given
				if (x,y) in line_tuples:
					new.append(1)
				else:
					new.append(0)
			alignment_matrix.append(new)
			new = []

		# get a line of words from the language 1 file
		lang1_line = lang1_infile.readline()
	#	print lang1_line
		# strip away the \n symbol
		lang1_line = lang1_line.rstrip()
		# split the line into a list of strings
		lang1_line_list = lang1_line.split()

		# get a line of words from the language 2 file
		lang2_line = lang2_infile.readline()
		# strip away the \n symbol
		lang2_line = lang2_line.rstrip()
		# split the line into a list of strings
		lang2_line_list = lang2_line.split()

		# create the table for the alignment matrix
		print('<h3>Table', line_count, '</h3>')
		print('<table border="1">')
		print('  <thead><tr><td></td>', end=' ')
		# print the top row of words
		for word in lang2_line_list:
			print('<td> <div class="vertical">', word, '</div></td>', end=' ')
		print('</tr></thead>')
		# print the matrix
		for row in range(0, max_x+1):
			print('  <tr><td>')
			# print a word of the vertical sentence
			if row < len(lang1_line_list):
				print(lang1_line_list[row], end=' ')
			print('</td>', end=' ')
			for col in range (0, max_y+1):
				# if this is an alignment cell
				if alignment_matrix[row][col] == 1:
					print('<td bgcolor="black">')
				else:
					print('<td>')
				# in any case print the closing tag
				print('</td>', end=' ')
			print('</tr>')
		print('</table>')

		## print an empty line
		print()

		## for restricting the number of tables
		if (line_count >= number_of_tables + start_line_num):
			break

	## close the files
	alignment_infile.close()
	lang1_infile.close()
	lang2_infile.close()

	print('</body>')
	print('</html>')

###############################

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("source_filename", type=str, help="Path to source file.")
	parser.add_argument("target_filename", type=str, help="Path to target file.")
	parser.add_argument("alignment_filename", type=str, help="Path to alignment file.")
	parser.add_argument("-l", "--start-line", type=int, help="The start line number in the alignment file.")
	parser.add_argument("-n", "--number-tables", type=int, help="The number of tables to be created")
	args = parser.parse_args()

	# python3 display_alignment_tables.py "/home/user/staehli/master_thesis/data/MuST-C/test.tc.en" "/home/user/staehli/master_thesis/data/MuST-C/test.tc.de" "/home/user/staehli/master_thesis/homophone_analysis/alignments/forward_src-ref.align"

	display_alignment_tables(args.source_filename, args.target_filename, args.alignment_filename)
if __name__ == "__main__":
	main()