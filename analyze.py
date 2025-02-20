from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path
import re
import pandas as pd
from datetime import datetime


@dataclass
class CSVEntry:
    """Repräsentiert eine einzelne Zeile aus der CSV-Datei"""
    response_time: float
    dns_dialup: float
    dns: float
    request_write: float
    response_delay: float
    response_read: float
    status_code: int
    offset: float


@dataclass
class Batch:
    """Repräsentiert einen Batch (1/4 der Anfragen eines Runs)"""
    entries: List[CSVEntry]
    batch_number: int  # 1-4
    batch_size: int = field(init=False)

    def __post_init__(self):
        self.batch_size = len(self.entries)

    @property
    def avg_response_time(self) -> float:
        return sum(entry.response_time for entry in self.entries) / len(self.entries)


@dataclass
class BatchGroup:
    """Repräsentiert alle 4 Batches eines Runs"""
    batches: List[Batch]
    run_number: int  # V in filename
    test_number: int  # W in filename

    @property
    def total_requests(self) -> int:
        return sum(batch.batch_size for batch in self.batches)



@dataclass
class Run:
    """Repräsentiert einen kompletten Run mit Szenario und Modus"""
    batch_groups: List[BatchGroup]
    scenario: int  # Y in path (1-3)
    mode: int  # Z in path (1-3)

    @property
    def scenario_name(self) -> str:
        scenarios = {
            1: "Wait 5 Seconds",
            2: "CPU intensive Matrix-Multiplication",
            3: "IO-heavy db app"
        }
        return scenarios.get(self.scenario, "Unknown")

    @property
    def mode_name(self) -> str:
        modes = {
            1: "Spring MVC",
            2: "Spring MVC with Virtual Threads",
            3: "Spring WebFlux"
        }
        return modes.get(self.mode, "Unknown")


@dataclass
class TestSuite:
    """Hauptklasse, die alle Runs pro PC verwaltet"""
    pc_runs: Dict[str, List[Run]] = field(default_factory=dict)

    def add_run(self, pc_name: str, run: Run):
        if pc_name not in self.pc_runs:
            self.pc_runs[pc_name] = []
        self.pc_runs[pc_name].append(run)


class CSVParser:
    """Hilfsklasse zum Parsen der CSV-Dateien und Erstellen der Objektstruktur"""

    @staticmethod
    def parse_csv_entry(row) -> CSVEntry:
        return CSVEntry(
            response_time=float(row['response-time']),
            dns_dialup=float(row['DNS+dialup']),
            dns=float(row['DNS']),
            request_write=float(row['Request-write']),
            response_delay=float(row['Response-delay']),
            response_read=float(row['Response-read']),
            status_code=int(row['status-code']),
            offset=float(row['offset'])
        )

    @staticmethod
    def parse_path_info(path: Path) -> tuple[str, int, int]:
        """Extrahiert PC, Szenario und Modus aus dem Pfad"""
        match = re.match(r'cnc(\w+)_(\d+)-(\d+)', path.parent.name)
        if not match:
            raise ValueError(f"Invalid path format: {path}")
        return match.group(1), int(match.group(2)), int(match.group(3))

    @staticmethod
    def parse_filename_info(filename: str) -> tuple[int, int]:
        """Extrahiert Run-Nummer und Test-Nummer aus dem Dateinamen"""
        match = re.match(r'result-\d+_\d+-(\d+)-(\d+)\.csv', filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        return int(match.group(1)), int(match.group(2))

    @classmethod
    def parse_file(cls, file_path: Path) -> tuple[str, Run]:
        """Parst eine einzelne CSV-Datei und erstellt die entsprechenden Objekte"""
        pc_name, scenario, mode = cls.parse_path_info(file_path)
        run_number, test_number = cls.parse_filename_info(file_path.name)

        # CSV einlesen
        df = pd.read_csv(file_path)
        entries = [cls.parse_csv_entry(row) for _, row in df.iterrows()]

        # Batches erstellen (4 gleich große Gruppen)
        batch_size = len(entries) // 4
        batches = []
        for i in range(4):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_entries = entries[start_idx:end_idx]
            batches.append(Batch(entries=batch_entries, batch_number=i + 1))

        # BatchGroup erstellen
        batch_group = BatchGroup(
            batches=batches,
            run_number=run_number,
            test_number=test_number
        )

        # Run erstellen
        run = Run(
            batch_groups=[batch_group],
            scenario=scenario,
            mode=mode
        )

        return pc_name, run


def process_directory(input_dir: str) -> TestSuite:
    """Verarbeitet alle CSV-Dateien in einem Verzeichnis und erstellt eine TestSuite"""
    test_suite = TestSuite()
    input_path = Path(input_dir)

    for csv_file in input_path.glob('**/result-*.csv'):
        try:
            pc_name, run = CSVParser.parse_file(csv_file)
            test_suite.add_run(pc_name, run)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    return test_suite


# Beispiel-Verwendung:
if __name__ == "__main__":
    test_suite = process_directory("input")

    # Beispiel für Zugriff auf die Daten:
    for pc_name, runs in test_suite.pc_runs.items():
        print(f"\nPC: {pc_name}")
        for run in runs:
            print(f"  Scenario: {run.scenario_name}")
            print(f"  Mode: {run.mode_name}")
            for batch_group in run.batch_groups:
                print(f"    Run {batch_group.run_number}, Test {batch_group.test_number}")
                print(f"    Total requests: {batch_group.total_requests}")
                for batch in batch_group.batches:
                    print(f"      Batch {batch.batch_number}: "
                          f"Avg response time: {batch.avg_response_time:.3f}ms")
