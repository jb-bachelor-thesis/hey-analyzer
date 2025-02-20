import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class CSVEntry:
    """Represents a single line from the CSV file"""
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
    """Represents a batch (1/4 of the requests of a run)"""
    entries: List[CSVEntry]
    batch_number: int  # 1-4
    batch_size: int = field(init=False)

    def __post_init__(self):
        self.batch_size = len(self.entries)

    @property
    def avg_response_time(self) -> float:
        return sum(entry.response_time for entry in self.entries) / len(self.entries)

    @property
    def requests_per_second(self) -> float:
        # Maximum duration in the batch divided by number of requests
        max_duration = max(entry.response_time for entry in self.entries)
        return self.batch_size / max_duration if max_duration > 0 else 0


@dataclass
class BatchGroup:
    """Represents all 4 batches of a run"""
    batches: List[Batch]
    run_number: int  # V in filename
    test_number: int  # W in filename

    @property
    def total_requests(self) -> int:
        return sum(batch.batch_size for batch in self.batches)

    @property
    def median_response_time(self) -> float:
        """Calculate median response time from batches 2, 3, and 4"""
        stable_batches = [b.avg_response_time for b in self.batches[1:]]  # Skip first batch
        return median(stable_batches)

    @property
    def median_requests_per_second(self) -> float:
        """Calculate median requests per second from batches 2, 3, and 4"""
        stable_batches = [b.requests_per_second for b in self.batches[1:]]  # Skip first batch
        return median(stable_batches)


@dataclass
class Run:
    """Represents a complete run with scenario and mode"""
    batch_groups: List[BatchGroup]
    scenario: int  # Y in path (1-3)
    mode: int  # Z in path (1-3)
    concurrent_users: int  # W from filename

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
            1: "Spring MVC Thread Pool",
            2: "Spring MVC Virtual Threads",
            3: "Spring WebFlux"
        }
        return modes.get(self.mode, "Unknown")

    @property
    def avg_response_time(self) -> float:
        """Average response time across all batch groups"""
        return median([bg.median_response_time for bg in self.batch_groups])

    @property
    def avg_requests_per_second(self) -> float:
        """Average requests per second across all batch groups"""
        return median([bg.median_requests_per_second for bg in self.batch_groups])


@dataclass
class TestSuite:
    """Main class managing all runs per PC"""
    pc_runs: Dict[str, List[Run]] = field(default_factory=dict)

    def add_run(self, pc_name: str, run: Run):
        if pc_name not in self.pc_runs:
            self.pc_runs[pc_name] = []
        self.pc_runs[pc_name].append(run)

    def get_scenario_data(self, scenario: int) -> Dict[int, Dict[str, List[float]]]:
        """Get aggregated data for a specific scenario across all PCs"""
        mode_data = {}
        for runs in self.pc_runs.values():
            scenario_runs = [r for r in runs if r.scenario == scenario]
            for run in scenario_runs:
                if run.concurrent_users not in mode_data:
                    mode_data[run.concurrent_users] = {
                        'response_times': {1: [], 2: [], 3: []},
                        'requests_per_second': {1: [], 2: [], 3: []}
                    }
                mode_data[run.concurrent_users]['response_times'][run.mode].append(run.avg_response_time)
                mode_data[run.concurrent_users]['requests_per_second'][run.mode].append(run.avg_requests_per_second)

        return mode_data


class CSVParser:
    """Helper class for parsing CSV files and creating object structure"""

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
        """Extract PC, scenario and mode from path"""
        match = re.match(r'cnc(\w+)_(\d+)-(\d+)', path.parent.name)
        if not match:
            raise ValueError(f"Invalid path format: {path}")
        return match.group(1), int(match.group(2)), int(match.group(3))

    @staticmethod
    def parse_filename_info(filename: str) -> tuple[int, int]:
        """Extract run number and test number from filename"""
        match = re.match(r'result-\d+_\d+-(\d+)-(\d+)\.csv', filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        return int(match.group(1)), int(match.group(2))

    @classmethod
    def parse_file(cls, file_path: Path) -> tuple[str, Run]:
        """Parse a single CSV file and create corresponding objects"""
        pc_name, scenario, mode = cls.parse_path_info(file_path)
        run_number, test_number = cls.parse_filename_info(file_path.name)

        # Read CSV
        df = pd.read_csv(file_path)
        entries = [cls.parse_csv_entry(row) for _, row in df.iterrows()]

        # Create batches (4 equal-sized groups)
        batch_size = len(entries) // 4
        batches = []
        for i in range(4):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_entries = entries[start_idx:end_idx]
            batches.append(Batch(entries=batch_entries, batch_number=i + 1))

        # Create BatchGroup
        batch_group = BatchGroup(
            batches=batches,
            run_number=run_number,
            test_number=test_number
        )

        # Create Run
        run = Run(
            batch_groups=[batch_group],
            scenario=scenario,
            mode=mode,
            concurrent_users=test_number
        )

        return pc_name, run


class ResultsGenerator:
    """Class for generating graphs and summary reports"""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_scenario_graphs(self, test_suite: TestSuite, scenario: int):
        """Generate response time and requests/sec graphs for a scenario."""
        scenario_data = test_suite.get_scenario_data(scenario)

        # Derive the scenario name
        scenario_name = Run([], scenario, 1, 1).scenario_name

        # Prepare data for plotting
        concurrent_users = sorted(scenario_data.keys())
        modes = [1, 2, 3]

        # Plot response times
        self._create_plot(
            scenario_data, concurrent_users, modes,
            'response_times', 'Average Response Time',
            f'response_time_scenario_{scenario}.png',
            'Response Time (ms)', scenario_name, log_scale=True
        )

        # Plot requests per second
        self._create_plot(
            scenario_data, concurrent_users, modes,
            'requests_per_second', 'Requests per Second',
            f'requests_per_second_scenario_{scenario}.png',
            'Requests/Second', scenario_name, log_scale=True
        )

    def _create_plot(self, data, x_values, modes, metric, title, filename, y_label, scenario_name, log_scale=False):
        plt.figure(figsize=(10, 6))

        colors = ['blue', 'red', 'green']
        labels = ['Thread Pool', 'Virtual Threads', 'WebFlux']
        markers = ['s', 'o', '^']

        for mode, color, label, marker in zip(modes, colors, labels, markers):
            y_values = []
            for x in x_values:
                values = data[x][metric][mode]
                avg = sum(values) / len(values) if values else 0
                y_values.append(avg)

            plt.plot(x_values, y_values, color=color, label=label,
                     marker=marker, linestyle='-', linewidth=2, markersize=8)

        plt.xlabel('Concurrent Users')
        plt.ylabel(y_label)
        plt.title(f'{title} - Scenario {scenario_name}')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()

        if log_scale:
            plt.yscale('log')
        plt.xscale('log', base=2)
        plt.xticks(x_values, [str(x) for x in x_values])

        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

    def generate_markdown_summary(self, test_suite: TestSuite):
        """Generate markdown summary of results"""
        summary = ["# Performance Test Results Summary\n"]

        for scenario in range(1, 4):
            scenario_data = test_suite.get_scenario_data(scenario)
            summary.append(f"## Scenario {scenario}: {Run([], scenario, 1, 1).scenario_name}\n")

            summary.append("### Response Times (ms)\n")
            summary.append("| Concurrent Users | Thread Pool | Virtual Threads | WebFlux |")
            summary.append("|-----------------|-------------|----------------|---------|")

            for users in sorted(scenario_data.keys()):
                values = []
                for mode in range(1, 4):
                    mode_values = scenario_data[users]['response_times'][mode]
                    avg = f"{sum(mode_values) / len(mode_values):.2f}" if mode_values else "N/A"
                    values.append(avg)
                summary.append(f"| {users} | {values[0]} | {values[1]} | {values[2]} |")

            summary.append("\n### Requests per Second\n")
            summary.append("| Concurrent Users | Thread Pool | Virtual Threads | WebFlux |")
            summary.append("|-----------------|-------------|----------------|---------|")

            for users in sorted(scenario_data.keys()):
                values = []
                for mode in range(1, 4):
                    mode_values = scenario_data[users]['requests_per_second'][mode]
                    avg = f"{sum(mode_values) / len(mode_values):.2f}" if mode_values else "N/A"
                    values.append(avg)
                summary.append(f"| {users} | {values[0]} | {values[1]} | {values[2]} |")

            summary.append("\n")

        with open(self.output_dir / "summary.md", "w") as f:
            f.write("\n".join(summary))


def process_directory(input_dir: str) -> TestSuite:
    """Process all CSV files in a directory and create a TestSuite"""
    test_suite = TestSuite()
    input_path = Path(input_dir)

    for csv_file in input_path.glob('**/result-*.csv'):
        try:
            pc_name, run = CSVParser.parse_file(csv_file)
            test_suite.add_run(pc_name, run)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")

    return test_suite


if __name__ == "__main__":
    # Create test suite from input directory
    test_suite = process_directory("input")

    # Generate results
    results_gen = ResultsGenerator()

    # Generate graphs for each scenario
    for scenario in range(1, 4):
        results_gen.generate_scenario_graphs(test_suite, scenario)

    # Generate markdown summary
    results_gen.generate_markdown_summary(test_suite)
