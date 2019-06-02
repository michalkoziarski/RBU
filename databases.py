import argparse
import datetime
import logging
import os
import pandas as pd
import sqlite3


RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'results')
PENDING_PATH = os.path.join(RESULTS_PATH, 'pending.db')
ACTIVE_PATH = os.path.join(RESULTS_PATH, 'active.db')
FINISHED_PATH = os.path.join(RESULTS_PATH, 'finished.db')


def add_to_pending(trial, check_presence=True):
    if check_presence:
        for database_path in [PENDING_PATH, ACTIVE_PATH, FINISHED_PATH]:
            if _select(trial, database_path=database_path) is not None:
                return False

    _insert(trial, database_path=PENDING_PATH)

    return True


def pull_pending():
    connection = _connect(database_path=PENDING_PATH, exclusive=True)
    trial = _select(connection=connection, fetch='one')

    if trial is not None:
        _insert(trial, database_path=ACTIVE_PATH)
        _delete(trial, connection=connection)

    connection.commit()
    connection.close()

    return trial


def submit_result(trial, scores):
    _delete(trial, database_path=ACTIVE_PATH)

    trial.update({'Scores': scores})

    _insert(trial, database_path=FINISHED_PATH)


def clear_active():
    trials = _select(database_path=ACTIVE_PATH, fetch='all')

    for trial in trials:
        _delete(trial, ACTIVE_PATH)
        _insert(trial, PENDING_PATH)


def export(path=None, use_timestamp=False, database_path=FINISHED_PATH):
    if path is None:
        if use_timestamp:
            timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
            path = os.path.join(RESULTS_PATH, 'results_%s.csv' % timestamp)
        else:
            path = os.path.join(RESULTS_PATH, 'results.csv')

    trials = _select(database_path=database_path, fetch='all')
    df = pd.DataFrame(trials, columns=_columns(score=(database_path == FINISHED_PATH)))
    df.to_csv(path, index=False)


def initialize():
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

    for path in [PENDING_PATH, ACTIVE_PATH, FINISHED_PATH]:
        if not os.path.exists(path):
            if path == FINISHED_PATH:
                columns = _columns(score=True)
            else:
                columns = _columns(score=False)

            columns = ['%s text' % column for column in columns]
            columns = ', '.join(columns)

            _execute('CREATE TABLE Trials (%s)' % columns, database_path=path)


def _select(trial=None, database_path=None, connection=None, fetch='one'):
    assert fetch in ['one', 'all']

    command = 'SELECT * FROM Trials'

    if trial is not None:
        command += ' %s' % _selector(trial)

    return _execute(command, database_path=database_path, connection=connection, fetch=fetch)


def _insert(trial, database_path=None, connection=None):
    _execute('INSERT INTO Trials (%s) VALUES (%s)' %
             (', '.join(trial.keys()), ', '.join(['"%s"' % value for value in trial.values()])),
             database_path=database_path, connection=connection)


def _delete(trial, database_path=None, connection=None):
    return _execute('DELETE FROM Trials %s' % _selector(trial), database_path=database_path, connection=connection)


def _execute(command, connection=None, database_path=None, fetch='none'):
    assert connection is not None or database_path is not None
    assert fetch in ['none', 'one', 'all']

    if connection is None:
        conn = _connect(database_path)
    else:
        conn = connection

    cursor = conn.cursor()
    cursor.execute(command)

    if fetch == 'one':
        result = cursor.fetchone()
    elif fetch == 'all':
        result = cursor.fetchall()
    else:
        result = None

    if connection is None:
        conn.commit()
        conn.close()

    return result


def _connect(database_path, exclusive=False, timeout=600.0):
    connection = sqlite3.connect(database_path, timeout=timeout)
    connection.row_factory = _dict_factory

    if exclusive:
        connection.isolation_level = 'EXCLUSIVE'
        connection.execute('BEGIN EXCLUSIVE')

    return connection


def _columns(score):
    columns = ['Algorithm', 'Parameters', 'Dataset', 'Fold', 'Description', 'Scores']

    if score:
        return columns
    else:
        return columns[:-1]


def _selector(trial):
    selector = 'WHERE '

    for k, v in trial.items():
        if k == 'Scores':
            continue

        selector += '%s="%s" AND ' % (k, v)

    selector = selector[:-5]

    return selector


def _dict_factory(cursor, row):
    d = {}

    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]

    return d


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument('-command', type=str, choices=['create', 'export', 'clear'])

    args = parser.parse_args()

    if args.command == 'create':
        logging.info('Creating databases...')

        initialize()
    elif args.command == 'export':
        logging.info('Exporting results...')

        export()
    elif args.command == 'clear':
        logging.info('Clearing active trials...')

        clear_active()
    else:
        raise NotImplementedError
