{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import re\n",
    "from sqlalchemy import create_engine\n",
    "import sqlalchemy as db\n",
    "from config import db_password"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "file_dir = 'C:/Users/12096/Desktop/BootCamp/08 Module DABC - ETL/Movies-ETL'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'C:/Users/12096/Desktop/BootCamp/08 Module DABC - ETL/Movies-ETL/wikipedia.movies.json'"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f'{file_dir}/wikipedia.movies.json'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(f'{file_dir}/wikipedia.movies.json', mode='r') as file:\n",
    "    wiki_movies_raw = json.load(file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "kaggle_metadata = pd.read_csv(f'{file_dir}/movies_metadata.csv')\n",
    "ratings = pd.read_csv(f'{file_dir}/ratings.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Convert wiki movies into a DataFrame\n",
    "wiki_movies_df = pd.DataFrame(wiki_movies_raw)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "7076"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "wiki_movies = [movie for movie in wiki_movies_raw\n",
    "               if ('Director' in movie or 'Directed by' in movie)\n",
    "                   and 'imdb_link' in movie\n",
    "                   and 'No. of episodes' not in movie]\n",
    "len(wiki_movies)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_movie(movie):\n",
    "    movie = dict(movie) #create a non-destructive copy\n",
    "    alt_titles = {}\n",
    "    # combine alternate titles into one list\n",
    "    for key in ['Also known as','Arabic','Cantonese','Chinese','French',\n",
    "                'Hangul','Hebrew','Hepburn','Japanese','Literally',\n",
    "                'Mandarin','McCune-Reischauer','Original title','Polish',\n",
    "                'Revised Romanization','Romanized','Russian',\n",
    "                'Simplified','Traditional','Yiddish']:\n",
    "        if key in movie:\n",
    "            alt_titles[key] = movie[key]\n",
    "            movie.pop(key)\n",
    "    if len(alt_titles) > 0:\n",
    "        movie['alt_titles'] = alt_titles\n",
    "\n",
    "    # merge column names\n",
    "   \n",
    "    def change_column_name(old_name, new_name):\n",
    "        if old_name in movie:\n",
    "            movie[new_name] = movie.pop(old_name)\n",
    "    change_column_name('Adaptation by', 'Writer(s)')\n",
    "    change_column_name('Country of origin', 'Country')\n",
    "    change_column_name('Directed by', 'Director')\n",
    "    change_column_name('Distributed by', 'Distributor')\n",
    "    change_column_name('Edited by', 'Editor(s)')\n",
    "    change_column_name('Length', 'Running time')\n",
    "    change_column_name('Original release', 'Release date')\n",
    "    change_column_name('Music by', 'Composer(s)')\n",
    "    change_column_name('Produced by', 'Producer(s)')\n",
    "    change_column_name('Producer', 'Producer(s)')\n",
    "    change_column_name('Productioncompanies ', 'Production company(s)')\n",
    "    change_column_name('Productioncompany ', 'Production company(s)')\n",
    "    change_column_name('Released', 'Release Date')\n",
    "    change_column_name('Release Date', 'Release date')\n",
    "    change_column_name('Screen story by', 'Writer(s)')\n",
    "    change_column_name('Screenplay by', 'Writer(s)')\n",
    "    change_column_name('Story by', 'Writer(s)')\n",
    "    change_column_name('Theme music composer', 'Composer(s)')\n",
    "    change_column_name('Written by', 'Writer(s)')\n",
    "    \n",
    "\n",
    "    return movie\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "clean_movies = [clean_movie(movie) for movie in wiki_movies]\n",
    "wiki_movies_df = pd.DataFrame(clean_movies)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\\d{7})')\n",
    "\n",
    "wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]\n",
    "wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Drop null values in Box Office column\n",
    "box_office = wiki_movies_df['Box office'].dropna() \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "form_one = r'\\$\\d+\\.?\\d*\\s*[mb]illion'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1544"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "form_two = r'\\$\\d{1,3}(?:,\\d{3})+'\n",
    "box_office.str.contains(form_two, flags=re.IGNORECASE).sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [],
   "source": [
    "matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)\n",
    "matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "box_office = box_office.str.replace(r'\\$.*[-—–](?![a-z])', '$', regex=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create the parse_dollars function to normalize the data across differing formats\n",
    "def parse_dollars(s):\n",
    "    # if s is not a string, return NaN\n",
    "    if type(s) != str:\n",
    "        return np.nan\n",
    "\n",
    "    # if input is of the form $###.# million\n",
    "    if re.match(r'\\$\\s*\\d+\\.?\\d*\\s*milli?on', s, flags=re.IGNORECASE):\n",
    "\n",
    "        # remove dollar sign and \" million\"\n",
    "        s = re.sub('\\$|\\s|[a-zA-Z]','', s)\n",
    "\n",
    "        # convert to float and multiply by a million\n",
    "        value = float(s) * 10**6\n",
    "\n",
    "        # return value\n",
    "        return value\n",
    "\n",
    "    # if input is of the form $###.# billion\n",
    "    elif re.match(r'\\$\\s*\\d+\\.?\\d*\\s*billi?on', s, flags=re.IGNORECASE):\n",
    "\n",
    "        # remove dollar sign and \" billion\"\n",
    "        s = re.sub('\\$|\\s|[a-zA-Z]','', s)\n",
    "\n",
    "        # convert to float and multiply by a billion\n",
    "        value = float(s) * 10**9\n",
    "\n",
    "        # return value\n",
    "        return value\n",
    "\n",
    "    # if input is of the form $###,###,###\n",
    "    elif re.match(r'\\$\\s*\\d{1,3}(?:[,\\.]\\d{3})+(?!\\s[mb]illion)', s, flags=re.IGNORECASE):\n",
    "\n",
    "        # remove dollar sign and commas\n",
    "        s = re.sub('\\$|,','', s)\n",
    "\n",
    "        # convert to float\n",
    "        value = float(s)\n",
    "\n",
    "        # return value\n",
    "        return value\n",
    "\n",
    "    # otherwise, return NaN\n",
    "    else:\n",
    "        return np.nan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "#extract the values from Box Office column and apply parse_dollars function\n",
    "wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Drop the Box Office column\n",
    "wiki_movies_df.drop('Box office', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create a variable for budget data\n",
    "budget = wiki_movies_df['Budget'].dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Convert list to strings\n",
    "budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Remove any value between $ and a hyphen, if error occurs skip\n",
    "try:\n",
    "    budget = budget.str.replace(r'\\$.*[-—–](?![a-z])', '$', regex=True)\n",
    "#If an error is experienced, skip movie\n",
    "except:\n",
    "    print(\"Budget error. Skipping...\")\n",
    "    pass    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "135                  USD$ 9 million\n",
       "136                         Unknown\n",
       "204     60 million Norwegian Kroner\n",
       "255                        $ 80,000\n",
       "351              US$ 65 million [1]\n",
       "                   ...             \n",
       "6821                  £12.9 million\n",
       "6843                      3.5 crore\n",
       "6895                        919,000\n",
       "6904                    $8.6 millon\n",
       "7070                   €4.3 million\n",
       "Name: Budget, Length: 76, dtype: object"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Use pattern matches to parse the budgets data\n",
    "matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)\n",
    "matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)\n",
    "budget[~matches_form_one & ~matches_form_two]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "135                  USD$ 9 million\n",
       "136                         Unknown\n",
       "204     60 million Norwegian Kroner\n",
       "255                        $ 80,000\n",
       "351                 US$ 65 million \n",
       "                   ...             \n",
       "6821                  £12.9 million\n",
       "6843                      3.5 crore\n",
       "6895                        919,000\n",
       "6904                    $8.6 millon\n",
       "7070                   €4.3 million\n",
       "Name: Budget, Length: 76, dtype: object"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Remove citation references\n",
    "budget = budget.str.replace(r'\\[\\d+\\]\\s*', '')\n",
    "budget[~matches_form_one & ~matches_form_two]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "#extract the values from Budget column and apply parse_dollars function, except errors\n",
    "try:\n",
    "    wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)\n",
    "except:\n",
    "    print(\"Budget error. Skipping...\")\n",
    "    pass      "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Drop the Budget column\n",
    "wiki_movies_df.drop('Budget', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Parse the release date - first make a variable go hold the release date data\n",
    "release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\\s[123]\\d,\\s\\d{4}'\n",
    "date_form_two = r'\\d{4}.[01]\\d.[123]\\d'\n",
    "date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\\s\\d{4}'\n",
    "date_form_four = r'\\d{4}'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert List to strings and apply forms\n",
    "try:\n",
    "    release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)\n",
    "except:\n",
    "    print(\"release date error. Skipping...\")\n",
    "    pass      "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Parse running time data\n",
    "running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "running_time_extract = running_time.str.extract(r'(\\d+)\\s*ho?u?r?s?\\s*(\\d*)|(\\d+)\\s*m')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "wiki_movies_df.drop('Running time', axis=1, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Kaggle Data Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)\n",
    "kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')\n",
    "kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "metadata": {},
   "outputs": [],
   "source": [
    "kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "metadata": {},
   "outputs": [],
   "source": [
    "ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Merge wiki_movies and kaggle_metadata\n",
    "movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>url</th>\n",
       "      <th>year</th>\n",
       "      <th>imdb_link</th>\n",
       "      <th>title_wiki</th>\n",
       "      <th>Based on</th>\n",
       "      <th>Starring</th>\n",
       "      <th>Cinematography</th>\n",
       "      <th>Release date</th>\n",
       "      <th>Country</th>\n",
       "      <th>Language</th>\n",
       "      <th>...</th>\n",
       "      <th>release_date_kaggle</th>\n",
       "      <th>revenue</th>\n",
       "      <th>runtime</th>\n",
       "      <th>spoken_languages</th>\n",
       "      <th>status</th>\n",
       "      <th>tagline</th>\n",
       "      <th>title_kaggle</th>\n",
       "      <th>video</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>vote_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>0 rows × 44 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [url, year, imdb_link, title_wiki, Based on, Starring, Cinematography, Release date, Country, Language, Director, Distributor, Editor(s), Composer(s), Producer(s), Production company(s), Writer(s), imdb_id, box_office, budget_wiki, release_date_wiki, running_time, belongs_to_collection, budget_kaggle, genres, homepage, id, original_language, original_title, overview, popularity, poster_path, production_companies, production_countries, release_date_kaggle, revenue, runtime, spoken_languages, status, tagline, title_kaggle, video, vote_average, vote_count]\n",
       "Index: []\n",
       "\n",
       "[0 rows x 44 columns]"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]\n",
    "except:\n",
    "    print(\"release date error. Skipping...\")\n",
    "    pass  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Int64Index([3607], dtype='int64')"
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Get the index of the row with bad merge\n",
    "movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "#drop the row with the bad merge  \n",
    "movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "English                             5479\n",
       "NaN                                  134\n",
       "(English, Spanish)                    68\n",
       "(English, French)                     35\n",
       "(English, Japanese)                   25\n",
       "                                    ... \n",
       "English subtitles                      1\n",
       "English, Arabic, Lakota                1\n",
       "(English, French, Chinese, Urdu)       1\n",
       "(English, Italian, Portuguese)         1\n",
       "(Chinese, English)                     1\n",
       "Name: Language, Length: 198, dtype: int64"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Competing data:\n",
    "# Wiki                     Movielens                Resolution\n",
    "#--------------------------------------------------------------------------\n",
    "# title_wiki               title_kaggle             drop wikipedia\n",
    "# running_time             runtime                  keep kaggle fill in zeros with wiki data\n",
    "# budget_wiki              budget_kaggle            keep kaggle fill in zeros with wiki data\n",
    "# box_office               revenue                  keep kaggle fill in zeros with wiki data\n",
    "# release_date_wiki        release_date_kaggle      Drop Wiki\n",
    "# Language                 original_language        Drop Wiki\n",
    "# Production company(s)    production_companies     Drop Wiki"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Drop the unneeded Wiki columns\n",
    "try:\n",
    "    movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)\n",
    "except:\n",
    "    print(\"No column to drop. Skipping...\")\n",
    "    pass      "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a function to fill in missing data on column pairs\n",
    "def fill_missing_kaggle_data(df, kaggle_column, wiki_column):\n",
    "    df[kaggle_column] = df.apply(\n",
    "        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]\n",
    "        , axis=1)\n",
    "    df.drop(columns=wiki_column, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>url</th>\n",
       "      <th>year</th>\n",
       "      <th>imdb_link</th>\n",
       "      <th>Based on</th>\n",
       "      <th>Starring</th>\n",
       "      <th>Cinematography</th>\n",
       "      <th>Release date</th>\n",
       "      <th>Country</th>\n",
       "      <th>Director</th>\n",
       "      <th>Distributor</th>\n",
       "      <th>...</th>\n",
       "      <th>release_date_kaggle</th>\n",
       "      <th>revenue</th>\n",
       "      <th>runtime</th>\n",
       "      <th>spoken_languages</th>\n",
       "      <th>status</th>\n",
       "      <th>tagline</th>\n",
       "      <th>title_kaggle</th>\n",
       "      <th>video</th>\n",
       "      <th>vote_average</th>\n",
       "      <th>vote_count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>https://en.wikipedia.org/wiki/The_Adventures_o...</td>\n",
       "      <td>1990</td>\n",
       "      <td>https://www.imdb.com/title/tt0098987/</td>\n",
       "      <td>[Characters, by Rex Weiner]</td>\n",
       "      <td>[Andrew Dice Clay, Wayne Newton, Priscilla Pre...</td>\n",
       "      <td>Oliver Wood</td>\n",
       "      <td>[July 11, 1990, (, 1990-07-11, )]</td>\n",
       "      <td>United States</td>\n",
       "      <td>Renny Harlin</td>\n",
       "      <td>20th Century Fox</td>\n",
       "      <td>...</td>\n",
       "      <td>1990-07-11</td>\n",
       "      <td>20423389.0</td>\n",
       "      <td>104.0</td>\n",
       "      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>Kojak. Columbo. Dirty Harry. Wimps.</td>\n",
       "      <td>The Adventures of Ford Fairlane</td>\n",
       "      <td>False</td>\n",
       "      <td>6.2</td>\n",
       "      <td>72.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>https://en.wikipedia.org/wiki/After_Dark,_My_S...</td>\n",
       "      <td>1990</td>\n",
       "      <td>https://www.imdb.com/title/tt0098994/</td>\n",
       "      <td>[the novel, After Dark, My Sweet, by, Jim Thom...</td>\n",
       "      <td>[Jason Patric, Rachel Ward, Bruce Dern, George...</td>\n",
       "      <td>Mark Plummer</td>\n",
       "      <td>[May 17, 1990, (, 1990-05-17, ), (Cannes Film ...</td>\n",
       "      <td>United States</td>\n",
       "      <td>James Foley</td>\n",
       "      <td>Avenue Pictures</td>\n",
       "      <td>...</td>\n",
       "      <td>1990-08-24</td>\n",
       "      <td>2700000.0</td>\n",
       "      <td>114.0</td>\n",
       "      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>All they risked was everything.</td>\n",
       "      <td>After Dark, My Sweet</td>\n",
       "      <td>False</td>\n",
       "      <td>6.5</td>\n",
       "      <td>17.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>https://en.wikipedia.org/wiki/Air_America_(film)</td>\n",
       "      <td>1990</td>\n",
       "      <td>https://www.imdb.com/title/tt0099005/</td>\n",
       "      <td>[Air America, by, Christopher Robbins]</td>\n",
       "      <td>[Mel Gibson, Robert Downey Jr., Nancy Travis, ...</td>\n",
       "      <td>Roger Deakins</td>\n",
       "      <td>[August 10, 1990, (, 1990-08-10, )]</td>\n",
       "      <td>United States</td>\n",
       "      <td>Roger Spottiswoode</td>\n",
       "      <td>TriStar Pictures</td>\n",
       "      <td>...</td>\n",
       "      <td>1990-08-10</td>\n",
       "      <td>33461269.0</td>\n",
       "      <td>112.0</td>\n",
       "      <td>[{'iso_639_1': 'en', 'name': 'English'}, {'iso...</td>\n",
       "      <td>Released</td>\n",
       "      <td>The few. The proud. The totally insane.</td>\n",
       "      <td>Air America</td>\n",
       "      <td>False</td>\n",
       "      <td>5.3</td>\n",
       "      <td>146.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>https://en.wikipedia.org/wiki/Alice_(1990_film)</td>\n",
       "      <td>1990</td>\n",
       "      <td>https://www.imdb.com/title/tt0099012/</td>\n",
       "      <td>NaN</td>\n",
       "      <td>[Alec Baldwin, Blythe Danner, Judy Davis, Mia ...</td>\n",
       "      <td>Carlo Di Palma</td>\n",
       "      <td>[December 25, 1990, (, 1990-12-25, )]</td>\n",
       "      <td>United States</td>\n",
       "      <td>Woody Allen</td>\n",
       "      <td>Orion Pictures</td>\n",
       "      <td>...</td>\n",
       "      <td>1990-12-25</td>\n",
       "      <td>7331647.0</td>\n",
       "      <td>102.0</td>\n",
       "      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Alice</td>\n",
       "      <td>False</td>\n",
       "      <td>6.3</td>\n",
       "      <td>57.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>https://en.wikipedia.org/wiki/Almost_an_Angel</td>\n",
       "      <td>1990</td>\n",
       "      <td>https://www.imdb.com/title/tt0099018/</td>\n",
       "      <td>NaN</td>\n",
       "      <td>[Paul Hogan, Elias Koteas, Linda Kozlowski]</td>\n",
       "      <td>Russell Boyd</td>\n",
       "      <td>December 19, 1990</td>\n",
       "      <td>US</td>\n",
       "      <td>John Cornell</td>\n",
       "      <td>Paramount Pictures</td>\n",
       "      <td>...</td>\n",
       "      <td>1990-12-21</td>\n",
       "      <td>6939946.0</td>\n",
       "      <td>95.0</td>\n",
       "      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>Who does he think he is?</td>\n",
       "      <td>Almost an Angel</td>\n",
       "      <td>False</td>\n",
       "      <td>5.6</td>\n",
       "      <td>23.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6047</td>\n",
       "      <td>https://en.wikipedia.org/wiki/A_Fantastic_Woman</td>\n",
       "      <td>2018</td>\n",
       "      <td>https://www.imdb.com/title/tt5639354/</td>\n",
       "      <td>NaN</td>\n",
       "      <td>[Daniela Vega, Francisco Reyes]</td>\n",
       "      <td>Benjamín Echazarreta</td>\n",
       "      <td>[12 February 2017, (, 2017-02-12, ), (, Berlin...</td>\n",
       "      <td>[Chile, Germany, Spain, United States, [2]]</td>\n",
       "      <td>Sebastián Lelio</td>\n",
       "      <td>[Participant Media (Chile), Piffl Medien (Germ...</td>\n",
       "      <td>...</td>\n",
       "      <td>2017-04-06</td>\n",
       "      <td>3700000.0</td>\n",
       "      <td>104.0</td>\n",
       "      <td>[{'iso_639_1': 'es', 'name': 'Español'}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>NaN</td>\n",
       "      <td>A Fantastic Woman</td>\n",
       "      <td>False</td>\n",
       "      <td>7.2</td>\n",
       "      <td>13.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6048</td>\n",
       "      <td>https://en.wikipedia.org/wiki/Permission_(film)</td>\n",
       "      <td>2018</td>\n",
       "      <td>https://www.imdb.com/title/tt5390066/</td>\n",
       "      <td>NaN</td>\n",
       "      <td>[Rebecca Hall, Dan Stevens, Morgan Spector, Fr...</td>\n",
       "      <td>Adam Bricker</td>\n",
       "      <td>[April 22, 2017, (, 2017-04-22, ), (, Tribeca ...</td>\n",
       "      <td>United States</td>\n",
       "      <td>Brian Crano</td>\n",
       "      <td>Good Deed Entertainment</td>\n",
       "      <td>...</td>\n",
       "      <td>2017-04-22</td>\n",
       "      <td>NaN</td>\n",
       "      <td>96.0</td>\n",
       "      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Permission</td>\n",
       "      <td>False</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6049</td>\n",
       "      <td>https://en.wikipedia.org/wiki/Loveless_(film)</td>\n",
       "      <td>2018</td>\n",
       "      <td>https://www.imdb.com/title/tt6304162/</td>\n",
       "      <td>NaN</td>\n",
       "      <td>[Maryana Spivak, Aleksey Rozin, Matvey Novikov...</td>\n",
       "      <td>Mikhail Krichman</td>\n",
       "      <td>[18 May 2017, (, 2017-05-18, ), (, Cannes, ), ...</td>\n",
       "      <td>[Russia, France, Belgium, Germany, [3]]</td>\n",
       "      <td>Andrey Zvyagintsev</td>\n",
       "      <td>[Sony Pictures Releasing, (Russia), [1]]</td>\n",
       "      <td>...</td>\n",
       "      <td>2017-06-01</td>\n",
       "      <td>4800000.0</td>\n",
       "      <td>128.0</td>\n",
       "      <td>[{'iso_639_1': 'ru', 'name': 'Pусский'}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Loveless</td>\n",
       "      <td>False</td>\n",
       "      <td>7.8</td>\n",
       "      <td>26.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6050</td>\n",
       "      <td>https://en.wikipedia.org/wiki/Gemini_(2017_film)</td>\n",
       "      <td>2018</td>\n",
       "      <td>https://www.imdb.com/title/tt5795086/</td>\n",
       "      <td>NaN</td>\n",
       "      <td>[Lola Kirke, Zoë Kravitz, Greta Lee, Michelle ...</td>\n",
       "      <td>Andrew Reed</td>\n",
       "      <td>[March 12, 2017, (, 2017-03-12, ), (, SXSW, ),...</td>\n",
       "      <td>United States</td>\n",
       "      <td>Aaron Katz</td>\n",
       "      <td>Neon</td>\n",
       "      <td>...</td>\n",
       "      <td>2017-03-12</td>\n",
       "      <td>200340.0</td>\n",
       "      <td>92.0</td>\n",
       "      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>\n",
       "      <td>Post Production</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Gemini</td>\n",
       "      <td>False</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6051</td>\n",
       "      <td>https://en.wikipedia.org/wiki/How_to_Talk_to_G...</td>\n",
       "      <td>2018</td>\n",
       "      <td>https://www.imdb.com/title/tt3859310/</td>\n",
       "      <td>[\", How to Talk to Girls at Parties, \", by, Ne...</td>\n",
       "      <td>[Elle Fanning, Alex Sharp, Nicole Kidman, Ruth...</td>\n",
       "      <td>Frank G. DeMarco</td>\n",
       "      <td>[May 21, 2017, (, 2017-05-21, ), (, Cannes, ),...</td>\n",
       "      <td>[United Kingdom, United States]</td>\n",
       "      <td>John Cameron Mitchell</td>\n",
       "      <td>[A24, StudioCanal UK]</td>\n",
       "      <td>...</td>\n",
       "      <td>2017-12-27</td>\n",
       "      <td>382053.0</td>\n",
       "      <td>102.0</td>\n",
       "      <td>[{'iso_639_1': 'en', 'name': 'English'}]</td>\n",
       "      <td>Released</td>\n",
       "      <td>Some girls are out of this world.</td>\n",
       "      <td>How to Talk to Girls at Parties</td>\n",
       "      <td>False</td>\n",
       "      <td>0.0</td>\n",
       "      <td>10.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>6051 rows × 37 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                    url  year  \\\n",
       "0     https://en.wikipedia.org/wiki/The_Adventures_o...  1990   \n",
       "1     https://en.wikipedia.org/wiki/After_Dark,_My_S...  1990   \n",
       "2      https://en.wikipedia.org/wiki/Air_America_(film)  1990   \n",
       "3       https://en.wikipedia.org/wiki/Alice_(1990_film)  1990   \n",
       "4         https://en.wikipedia.org/wiki/Almost_an_Angel  1990   \n",
       "...                                                 ...   ...   \n",
       "6047    https://en.wikipedia.org/wiki/A_Fantastic_Woman  2018   \n",
       "6048    https://en.wikipedia.org/wiki/Permission_(film)  2018   \n",
       "6049      https://en.wikipedia.org/wiki/Loveless_(film)  2018   \n",
       "6050   https://en.wikipedia.org/wiki/Gemini_(2017_film)  2018   \n",
       "6051  https://en.wikipedia.org/wiki/How_to_Talk_to_G...  2018   \n",
       "\n",
       "                                  imdb_link  \\\n",
       "0     https://www.imdb.com/title/tt0098987/   \n",
       "1     https://www.imdb.com/title/tt0098994/   \n",
       "2     https://www.imdb.com/title/tt0099005/   \n",
       "3     https://www.imdb.com/title/tt0099012/   \n",
       "4     https://www.imdb.com/title/tt0099018/   \n",
       "...                                     ...   \n",
       "6047  https://www.imdb.com/title/tt5639354/   \n",
       "6048  https://www.imdb.com/title/tt5390066/   \n",
       "6049  https://www.imdb.com/title/tt6304162/   \n",
       "6050  https://www.imdb.com/title/tt5795086/   \n",
       "6051  https://www.imdb.com/title/tt3859310/   \n",
       "\n",
       "                                               Based on  \\\n",
       "0                           [Characters, by Rex Weiner]   \n",
       "1     [the novel, After Dark, My Sweet, by, Jim Thom...   \n",
       "2                [Air America, by, Christopher Robbins]   \n",
       "3                                                   NaN   \n",
       "4                                                   NaN   \n",
       "...                                                 ...   \n",
       "6047                                                NaN   \n",
       "6048                                                NaN   \n",
       "6049                                                NaN   \n",
       "6050                                                NaN   \n",
       "6051  [\", How to Talk to Girls at Parties, \", by, Ne...   \n",
       "\n",
       "                                               Starring        Cinematography  \\\n",
       "0     [Andrew Dice Clay, Wayne Newton, Priscilla Pre...           Oliver Wood   \n",
       "1     [Jason Patric, Rachel Ward, Bruce Dern, George...          Mark Plummer   \n",
       "2     [Mel Gibson, Robert Downey Jr., Nancy Travis, ...         Roger Deakins   \n",
       "3     [Alec Baldwin, Blythe Danner, Judy Davis, Mia ...        Carlo Di Palma   \n",
       "4           [Paul Hogan, Elias Koteas, Linda Kozlowski]          Russell Boyd   \n",
       "...                                                 ...                   ...   \n",
       "6047                    [Daniela Vega, Francisco Reyes]  Benjamín Echazarreta   \n",
       "6048  [Rebecca Hall, Dan Stevens, Morgan Spector, Fr...          Adam Bricker   \n",
       "6049  [Maryana Spivak, Aleksey Rozin, Matvey Novikov...      Mikhail Krichman   \n",
       "6050  [Lola Kirke, Zoë Kravitz, Greta Lee, Michelle ...           Andrew Reed   \n",
       "6051  [Elle Fanning, Alex Sharp, Nicole Kidman, Ruth...      Frank G. DeMarco   \n",
       "\n",
       "                                           Release date  \\\n",
       "0                     [July 11, 1990, (, 1990-07-11, )]   \n",
       "1     [May 17, 1990, (, 1990-05-17, ), (Cannes Film ...   \n",
       "2                   [August 10, 1990, (, 1990-08-10, )]   \n",
       "3                 [December 25, 1990, (, 1990-12-25, )]   \n",
       "4                                     December 19, 1990   \n",
       "...                                                 ...   \n",
       "6047  [12 February 2017, (, 2017-02-12, ), (, Berlin...   \n",
       "6048  [April 22, 2017, (, 2017-04-22, ), (, Tribeca ...   \n",
       "6049  [18 May 2017, (, 2017-05-18, ), (, Cannes, ), ...   \n",
       "6050  [March 12, 2017, (, 2017-03-12, ), (, SXSW, ),...   \n",
       "6051  [May 21, 2017, (, 2017-05-21, ), (, Cannes, ),...   \n",
       "\n",
       "                                          Country               Director  \\\n",
       "0                                   United States           Renny Harlin   \n",
       "1                                   United States            James Foley   \n",
       "2                                   United States     Roger Spottiswoode   \n",
       "3                                   United States            Woody Allen   \n",
       "4                                              US           John Cornell   \n",
       "...                                           ...                    ...   \n",
       "6047  [Chile, Germany, Spain, United States, [2]]        Sebastián Lelio   \n",
       "6048                                United States            Brian Crano   \n",
       "6049      [Russia, France, Belgium, Germany, [3]]     Andrey Zvyagintsev   \n",
       "6050                                United States             Aaron Katz   \n",
       "6051              [United Kingdom, United States]  John Cameron Mitchell   \n",
       "\n",
       "                                            Distributor  ...  \\\n",
       "0                                      20th Century Fox  ...   \n",
       "1                                       Avenue Pictures  ...   \n",
       "2                                      TriStar Pictures  ...   \n",
       "3                                        Orion Pictures  ...   \n",
       "4                                    Paramount Pictures  ...   \n",
       "...                                                 ...  ...   \n",
       "6047  [Participant Media (Chile), Piffl Medien (Germ...  ...   \n",
       "6048                            Good Deed Entertainment  ...   \n",
       "6049           [Sony Pictures Releasing, (Russia), [1]]  ...   \n",
       "6050                                               Neon  ...   \n",
       "6051                              [A24, StudioCanal UK]  ...   \n",
       "\n",
       "     release_date_kaggle     revenue runtime  \\\n",
       "0             1990-07-11  20423389.0   104.0   \n",
       "1             1990-08-24   2700000.0   114.0   \n",
       "2             1990-08-10  33461269.0   112.0   \n",
       "3             1990-12-25   7331647.0   102.0   \n",
       "4             1990-12-21   6939946.0    95.0   \n",
       "...                  ...         ...     ...   \n",
       "6047          2017-04-06   3700000.0   104.0   \n",
       "6048          2017-04-22         NaN    96.0   \n",
       "6049          2017-06-01   4800000.0   128.0   \n",
       "6050          2017-03-12    200340.0    92.0   \n",
       "6051          2017-12-27    382053.0   102.0   \n",
       "\n",
       "                                       spoken_languages           status  \\\n",
       "0              [{'iso_639_1': 'en', 'name': 'English'}]         Released   \n",
       "1              [{'iso_639_1': 'en', 'name': 'English'}]         Released   \n",
       "2     [{'iso_639_1': 'en', 'name': 'English'}, {'iso...         Released   \n",
       "3              [{'iso_639_1': 'en', 'name': 'English'}]         Released   \n",
       "4              [{'iso_639_1': 'en', 'name': 'English'}]         Released   \n",
       "...                                                 ...              ...   \n",
       "6047           [{'iso_639_1': 'es', 'name': 'Español'}]         Released   \n",
       "6048           [{'iso_639_1': 'en', 'name': 'English'}]         Released   \n",
       "6049           [{'iso_639_1': 'ru', 'name': 'Pусский'}]         Released   \n",
       "6050           [{'iso_639_1': 'en', 'name': 'English'}]  Post Production   \n",
       "6051           [{'iso_639_1': 'en', 'name': 'English'}]         Released   \n",
       "\n",
       "                                      tagline  \\\n",
       "0         Kojak. Columbo. Dirty Harry. Wimps.   \n",
       "1             All they risked was everything.   \n",
       "2     The few. The proud. The totally insane.   \n",
       "3                                         NaN   \n",
       "4                    Who does he think he is?   \n",
       "...                                       ...   \n",
       "6047                                      NaN   \n",
       "6048                                      NaN   \n",
       "6049                                      NaN   \n",
       "6050                                      NaN   \n",
       "6051        Some girls are out of this world.   \n",
       "\n",
       "                         title_kaggle  video vote_average  vote_count  \n",
       "0     The Adventures of Ford Fairlane  False          6.2        72.0  \n",
       "1                After Dark, My Sweet  False          6.5        17.0  \n",
       "2                         Air America  False          5.3       146.0  \n",
       "3                               Alice  False          6.3        57.0  \n",
       "4                     Almost an Angel  False          5.6        23.0  \n",
       "...                               ...    ...          ...         ...  \n",
       "6047                A Fantastic Woman  False          7.2        13.0  \n",
       "6048                       Permission  False          0.0         1.0  \n",
       "6049                         Loveless  False          7.8        26.0  \n",
       "6050                           Gemini  False          0.0         0.0  \n",
       "6051  How to Talk to Girls at Parties  False          0.0        10.0  \n",
       "\n",
       "[6051 rows x 37 columns]"
      ]
     },
     "execution_count": 66,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Run the function on columns keep kaggle and fillin zeros with wiki\n",
    "try:\n",
    "    fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')\n",
    "except:\n",
    "    runtime = 0    \n",
    "try:    \n",
    "    fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')\n",
    "except:\n",
    "    budget_kaggle = 0      \n",
    "try:    \n",
    "    fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')\n",
    "except:\n",
    "    revenue = 0      \n",
    "movies_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False    6051\n",
       "Name: video, dtype: int64"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Drop video column\n",
    "movies_df['video'].value_counts(dropna=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Reorder the data to make easier to read\n",
    "movies_df = movies_df[['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',\n",
    "                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',\n",
    "                       'genres','original_language','overview','spoken_languages','Country',\n",
    "                       'production_companies','production_countries','Distributor',\n",
    "                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'\n",
    "                      ]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\12096\\Anaconda3\\lib\\site-packages\\pandas\\core\\frame.py:4223: SettingWithCopyWarning: \n",
      "A value is trying to be set on a copy of a slice from a DataFrame\n",
      "\n",
      "See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
      "  return super().rename(**kwargs)\n"
     ]
    }
   ],
   "source": [
    "#Rename columns for consistency\n",
    "movies_df.rename({'id':'kaggle_id',\n",
    "                  'title_kaggle':'title',\n",
    "                  'url':'wikipedia_url',\n",
    "                  'budget_kaggle':'budget',\n",
    "                  'release_date_kaggle':'release_date',\n",
    "                  'Country':'country',\n",
    "                  'Distributor':'distributor',\n",
    "                  'Producer(s)':'producers',\n",
    "                  'Director':'director',\n",
    "                  'Starring':'starring',\n",
    "                  'Cinematography':'cinematography',\n",
    "                  'Editor(s)':'editors',\n",
    "                  'Writer(s)':'writers',\n",
    "                  'Composer(s)':'composers',\n",
    "                  'Based on':'based_on'\n",
    "                 }, axis='columns', inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>imdb_id</th>\n",
       "      <th>kaggle_id</th>\n",
       "      <th>title</th>\n",
       "      <th>original_title</th>\n",
       "      <th>tagline</th>\n",
       "      <th>belongs_to_collection</th>\n",
       "      <th>wikipedia_url</th>\n",
       "      <th>imdb_link</th>\n",
       "      <th>runtime</th>\n",
       "      <th>budget</th>\n",
       "      <th>...</th>\n",
       "      <th>production_countries</th>\n",
       "      <th>distributor</th>\n",
       "      <th>producers</th>\n",
       "      <th>director</th>\n",
       "      <th>starring</th>\n",
       "      <th>cinematography</th>\n",
       "      <th>editors</th>\n",
       "      <th>writers</th>\n",
       "      <th>composers</th>\n",
       "      <th>based_on</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>tt0098987</td>\n",
       "      <td>9548</td>\n",
       "      <td>The Adventures of Ford Fairlane</td>\n",
       "      <td>The Adventures of Ford Fairlane</td>\n",
       "      <td>Kojak. Columbo. Dirty Harry. Wimps.</td>\n",
       "      <td>NaN</td>\n",
       "      <td>https://en.wikipedia.org/wiki/The_Adventures_o...</td>\n",
       "      <td>https://www.imdb.com/title/tt0098987/</td>\n",
       "      <td>104.0</td>\n",
       "      <td>49000000.0</td>\n",
       "      <td>...</td>\n",
       "      <td>[{'iso_3166_1': 'US', 'name': 'United States o...</td>\n",
       "      <td>20th Century Fox</td>\n",
       "      <td>[Steve Perry, Joel Silver]</td>\n",
       "      <td>Renny Harlin</td>\n",
       "      <td>[Andrew Dice Clay, Wayne Newton, Priscilla Pre...</td>\n",
       "      <td>Oliver Wood</td>\n",
       "      <td>Michael Tronick</td>\n",
       "      <td>[David Arnott, James Cappe]</td>\n",
       "      <td>[Cliff Eidelman, Yello]</td>\n",
       "      <td>[Characters, by Rex Weiner]</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     imdb_id  kaggle_id                            title  \\\n",
       "0  tt0098987       9548  The Adventures of Ford Fairlane   \n",
       "\n",
       "                    original_title                              tagline  \\\n",
       "0  The Adventures of Ford Fairlane  Kojak. Columbo. Dirty Harry. Wimps.   \n",
       "\n",
       "  belongs_to_collection                                      wikipedia_url  \\\n",
       "0                   NaN  https://en.wikipedia.org/wiki/The_Adventures_o...   \n",
       "\n",
       "                               imdb_link  runtime      budget  ...  \\\n",
       "0  https://www.imdb.com/title/tt0098987/    104.0  49000000.0  ...   \n",
       "\n",
       "                                production_countries       distributor  \\\n",
       "0  [{'iso_3166_1': 'US', 'name': 'United States o...  20th Century Fox   \n",
       "\n",
       "                    producers      director  \\\n",
       "0  [Steve Perry, Joel Silver]  Renny Harlin   \n",
       "\n",
       "                                            starring cinematography  \\\n",
       "0  [Andrew Dice Clay, Wayne Newton, Priscilla Pre...    Oliver Wood   \n",
       "\n",
       "           editors                      writers                composers  \\\n",
       "0  Michael Tronick  [David Arnott, James Cappe]  [Cliff Eidelman, Yello]   \n",
       "\n",
       "                      based_on  \n",
       "0  [Characters, by Rex Weiner]  \n",
       "\n",
       "[1 rows x 31 columns]"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "movies_df.head(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Group and count movieID and rating\n",
    "rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {},
   "outputs": [],
   "source": [
    "rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \\\n",
    "                .rename({'userId':'count'}, axis=1) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Pivot the dat so movieID is the index\n",
    "rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \\\n",
    "                .rename({'userId':'count'}, axis=1) \\\n",
    "                .pivot(index='movieId',columns='rating', values='count')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Rename columns\n",
    "rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Merge ratings with movie data\n",
    "movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Fill in missing values\n",
    "movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "#http://127.0.0.1:50575/browser/\n",
    "\n",
    "db_string = f\"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    "engine = create_engine(db_string)\n",
    "connection = engine.connect()\n",
    "metadata = db.MetaData()\n",
    "movies = db.Table('movies', metadata, autoload=True, autoload_with=engine)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<sqlalchemy.engine.result.ResultProxy at 0x1f28eff3b88>"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Delete contents from movies and ratings tables\n",
    "connection.execute(\"DELETE FROM movies\")\n",
    "connection.execute(\"DELETE FROM ratings\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Update movies table data\n",
    "movies_df.to_sql(name='movies', con=engine, if_exists='append')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "importing rows 0 to 1000000...Done. 69.65316438674927 total seconds elapsed\n",
      "importing rows 1000000 to 2000000...Done. 133.89628911018372 total seconds elapsed\n",
      "importing rows 2000000 to 3000000...Done. 198.81064081192017 total seconds elapsed\n",
      "importing rows 3000000 to 4000000...Done. 266.10675740242004 total seconds elapsed\n",
      "importing rows 4000000 to 5000000...Done. 330.76396894454956 total seconds elapsed\n",
      "importing rows 5000000 to 6000000...Done. 396.6992816925049 total seconds elapsed\n",
      "importing rows 6000000 to 7000000...Done. 462.03300404548645 total seconds elapsed\n",
      "importing rows 7000000 to 8000000...Done. 527.5501744747162 total seconds elapsed\n",
      "importing rows 8000000 to 9000000...Done. 593.1569533348083 total seconds elapsed\n",
      "importing rows 9000000 to 10000000...Done. 657.9639961719513 total seconds elapsed\n",
      "importing rows 10000000 to 11000000...Done. 722.8261449337006 total seconds elapsed\n",
      "importing rows 11000000 to 12000000...Done. 788.1856260299683 total seconds elapsed\n",
      "importing rows 12000000 to 13000000...Done. 853.1969289779663 total seconds elapsed\n",
      "importing rows 13000000 to 14000000...Done. 919.3350238800049 total seconds elapsed\n",
      "importing rows 14000000 to 15000000...Done. 986.4298536777496 total seconds elapsed\n",
      "importing rows 15000000 to 16000000...Done. 1054.1889324188232 total seconds elapsed\n",
      "importing rows 16000000 to 17000000...Done. 1124.8255167007446 total seconds elapsed\n",
      "importing rows 17000000 to 18000000...Done. 1189.62189412117 total seconds elapsed\n",
      "importing rows 18000000 to 19000000...Done. 1255.165191411972 total seconds elapsed\n",
      "importing rows 19000000 to 20000000...Done. 1320.3231985569 total seconds elapsed\n",
      "importing rows 20000000 to 21000000...Done. 1385.794261932373 total seconds elapsed\n",
      "importing rows 21000000 to 22000000...Done. 1451.2313690185547 total seconds elapsed\n",
      "importing rows 22000000 to 23000000...Done. 1519.205311536789 total seconds elapsed\n",
      "importing rows 23000000 to 24000000...Done. 1585.7806074619293 total seconds elapsed\n",
      "importing rows 24000000 to 25000000...Done. 1653.6578595638275 total seconds elapsed\n",
      "importing rows 25000000 to 26000000...Done. 1719.585130929947 total seconds elapsed\n",
      "importing rows 26000000 to 26024289...Done. 1721.010621547699 total seconds elapsed\n"
     ]
    }
   ],
   "source": [
    "# create a variable for the number of rows imported\n",
    "rows_imported = 0\n",
    "# get the start_time from time.time()\n",
    "start_time = time.time()\n",
    "\n",
    "for data in pd.read_csv(f'{file_dir}/ratings.csv', chunksize=1000000):\n",
    "\n",
    "    # print out the range of rows that are being imported\n",
    "    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')\n",
    "\n",
    "    data.to_sql(name='ratings', con=engine, if_exists='append')\n",
    "\n",
    "    # increment the number of rows imported by the size of 'data'\n",
    "    rows_imported += len(data)\n",
    "\n",
    "    # print that the rows have finished importing\n",
    "    print(f'Done. {time.time() - start_time} total seconds elapsed')\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "raw",
   "metadata": {},
   "source": []
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
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
