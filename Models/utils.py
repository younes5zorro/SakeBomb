import os
import sys, getopt

def ensure_dir(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

def arguments(argv):
   inputfile = ''

   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile="])
   except getopt.GetoptError:

       print ("test.py -i <inputfile>")
       sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <inputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg

   print ('Input file is "', inputfile)

   return inputfile
