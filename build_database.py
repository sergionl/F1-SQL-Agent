"""Construye la base de datos SQLite del proyecto a partir del CSV de resultados de F1.

Uso:
    python build_database.py
    python build_database.py --csv data/races.csv --db agente_sql_f1.db --verify
"""

import argparse
import sqlite3

import pandas as pd
from sqlalchemy import create_engine

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"Datos cargados: {len(df)} filas desde {csv_path}")
    return df


def clean_data(df_original: pd.DataFrame) -> pd.DataFrame:
    df = df_original.copy()

    df = df.drop(['Q1', 'Q2', 'Q3'], axis=1)
    df = df.drop('HeadshotUrl', axis=1)
    df = df.drop('CountryCode', axis=1)
    df['Time'] = df['Time'].fillna('DNF')

    df = df.dropna(subset=["BroadcastName"])

    df['Position'] = df['Position'].astype('int64')
    df['GridPosition'] = df['GridPosition'].astype('int64')
    df['Points'] = df['Points'].astype('int64')
    df['Laps'] = df['Laps'].astype('int64')

    df['Time'] = df['Time'].apply(lambda x: x.split(" ")[-1] if x != 'DNF' else x)

    df['Race'] = df['year'].astype(str) + ' ' + df['raceName'].astype(str)
    df = df.drop('year', axis=1)
    df = df.drop('raceName', axis=1)

    duplicados = df.duplicated().sum()
    if duplicados:
        print(f"Registros duplicados eliminados: {duplicados}")
    df = df.drop_duplicates()

    print(f"Datos limpios: {len(df)} filas")
    return df


def build_tables(df_procesado: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_conductores = df_procesado[['DriverNumber', 'FullName', 'TeamName']].drop_duplicates().reset_index(drop=True)
    df_conductores.insert(0, 'conductor_id', range(1, len(df_conductores) + 1))

    df_resultados = pd.merge(df_procesado, df_conductores, on=['FullName', 'TeamName'])
    df_resultados = df_resultados[['conductor_id', 'Position', 'GridPosition', 'Time', 'Status', 'Points', 'Laps', 'Race']]

    return df_conductores, df_resultados


def add_features(df_resultados: pd.DataFrame) -> pd.DataFrame:
    df_resultados = df_resultados.copy()

    df_resultados['PositionDifference'] = df_resultados['GridPosition'] - df_resultados['Position']

    df_resultados['TimeTd'] = pd.to_timedelta(df_resultados['Time'], errors='coerce')
    leader_times = df_resultados[df_resultados['Position'] == 1].set_index('Race')['TimeTd']

    def format_total_time(row):
        if pd.isna(row['TimeTd']):
            return "DNF"
        total = row['TimeTd'] if row['Position'] == 1 else leader_times[row['Race']] + row['TimeTd']
        ts = total.total_seconds()
        hours, remainder = divmod(ts, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"

    df_resultados['TotalRaceTime'] = df_resultados.apply(format_total_time, axis=1)
    df_resultados = df_resultados.drop(columns=['TimeTd'])

    return df_resultados


def save_to_db(df_conductores: pd.DataFrame, df_resultados: pd.DataFrame, db_path: str) -> None:
    engine = create_engine(f'sqlite:///{db_path}')
    df_conductores.to_sql('conductores', engine, if_exists='replace', index=False)
    df_resultados.to_sql('resultados', engine, if_exists='replace', index=False)
    print(f"Base de datos creada exitosamente en: {db_path}")
    print(f"  Tabla 'conductores': {len(df_conductores)} registros")
    print(f"  Tabla 'resultados': {len(df_resultados)} registros")


def verify_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)

    print("\nTotal de registros en la base de datos:")
    print("Tabla conductores:")
    print(pd.read_sql_query("SELECT COUNT(*) as total FROM conductores", conn))
    print("Tabla resultados:")
    print(pd.read_sql_query("SELECT COUNT(*) as total FROM resultados", conn))

    print("\nPrimeras 5 filas:")
    print("Tabla conductores:")
    print(pd.read_sql_query("SELECT * FROM conductores LIMIT 5", conn))
    print("Tabla resultados:")
    print(pd.read_sql_query("SELECT * FROM resultados LIMIT 5", conn))

    print("\nAgregación de prueba:")
    print("Points:", pd.read_sql_query("SELECT AVG(Points) as promedio FROM resultados", conn).iloc[0, 0])
    print("Laps:", pd.read_sql_query("SELECT AVG(Laps) as promedio FROM resultados", conn).iloc[0, 0])

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Construye agente_sql_f1.db a partir del CSV de resultados de F1")
    parser.add_argument('--csv', default='data/races.csv', help='Ruta al CSV de origen (default: data/races.csv)')
    parser.add_argument('--db', default='agente_sql_f1.db', help='Ruta de la base de datos SQLite a generar')
    parser.add_argument('--verify', action='store_true', help='Ejecuta consultas de verificación tras la carga')
    args = parser.parse_args()

    df_original = load_data(args.csv)
    df_procesado = clean_data(df_original)
    df_conductores, df_resultados = build_tables(df_procesado)
    df_resultados = add_features(df_resultados)
    save_to_db(df_conductores, df_resultados, args.db)

    if args.verify:
        verify_db(args.db)


if __name__ == '__main__':
    main()
