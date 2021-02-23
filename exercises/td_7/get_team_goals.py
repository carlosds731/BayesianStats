def get_team_goals(team_name):
    """
        Returns an array with the number of goals the given team
        scored on every fixture.
    :param team_name: Name of the team. It should one of the following values
        ['Monaco', 'Marseille', 'Angers', 'Brest', 'Dijon', 'Montpellier',
         'Nice', 'Lille', 'Strasbourg', 'Paris SG', 'Lyon', 'Nantes',
          'Amiens', 'Bordeaux', 'Metz', 'Nimes', 'Toulouse', 'St Etienne',
          'Reims', 'Rennes']
    :type team_name: str
    :return:
    :rtype: list
    """
    import pandas
    # Store the raw data
    raw_data = pandas.DataFrame(pandas.read_csv('F1.csv'), columns=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'])
    home_results = raw_data.groupby(['HomeTeam']).get_group(team_name)
    away_results = raw_data.groupby(['AwayTeam']).get_group(team_name)
    home_results.insert(1, '_TeamGoals', home_results['FTHG'])
    away_results.insert(1, '_TeamGoals', away_results['FTAG'])
    # All results sorted by index
    all_results = home_results.append(away_results).sort_index()
    return all_results['_TeamGoals'].values
