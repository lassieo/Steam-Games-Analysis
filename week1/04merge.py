{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'march25' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m/Users/lushenasada/Documents/GitHub/Steam-Games-Analysis/04merge.ipynb Cell 1\u001b[0m line \u001b[0;36m1\n\u001b[0;32m----> <a href='vscode-notebook-cell:/Users/lushenasada/Documents/GitHub/Steam-Games-Analysis/04merge.ipynb#W0sZmlsZQ%3D%3D?line=0'>1</a>\u001b[0m merged \u001b[39m=\u001b[39m march25\u001b[39m.\u001b[39mcopy()\n\u001b[1;32m      <a href='vscode-notebook-cell:/Users/lushenasada/Documents/GitHub/Steam-Games-Analysis/04merge.ipynb#W0sZmlsZQ%3D%3D?line=1'>2</a>\u001b[0m merged\u001b[39m.\u001b[39mto_csv(\u001b[39m\"\u001b[39m\u001b[39mmerged_raw.csv\u001b[39m\u001b[39m\"\u001b[39m, index\u001b[39m=\u001b[39m\u001b[39mFalse\u001b[39;00m)\n",
      "\u001b[0;31mNameError\u001b[0m: name 'march25' is not defined"
     ]
    }
   ],
   "source": [
    "merged = march25.copy()\n",
    "merged.to_csv(\"merged_raw.csv\", index=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.1"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
