"""
OpenDSS file parser for SMART-DS data.
Parses NREL's SMART-DS distribution system models.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import re
import numpy as np


@dataclass
class DSSBus:
    """OpenDSS Bus definition."""
    name: str
    base_kv: float = 0.0
    x: float = 0.0
    y: float = 0.0
    phases: int = 3
    kv_ln: float = 0.0

    def __post_init__(self):
        if self.kv_ln == 0 and self.base_kv > 0:
            self.kv_ln = self.base_kv / np.sqrt(3)


@dataclass
class DSSLine:
    """OpenDSS Line definition."""
    name: str
    bus1: str
    bus2: str
    length: float = 1.0
    units: str = "km"
    r1: float = 0.0
    x1: float = 0.0
    r0: float = 0.0
    x0: float = 0.0
    c1: float = 0.0
    c0: float = 0.0
    normamps: float = 400.0
    emergamps: float = 600.0
    linecode: str = ""
    geometry: str = ""
    phases: int = 3
    enabled: bool = True


@dataclass
class DSSLoad:
    """OpenDSS Load definition."""
    name: str
    bus1: str
    phases: int = 3
    kv: float = 0.0
    kw: float = 0.0
    kvar: float = 0.0
    pf: float = 0.95
    model: int = 1
    conn: str = "wye"
    enabled: bool = True


@dataclass
class DSSGenerator:
    """OpenDSS Generator or PVSystem definition."""
    name: str
    bus1: str
    phases: int = 3
    kv: float = 0.0
    kw: float = 0.0
    kvar: float = 0.0
    pf: float = 1.0
    gen_type: str = "pv"  # pv, wind, diesel, etc.
    enabled: bool = True


@dataclass
class DSSTransformer:
    """OpenDSS Transformer definition."""
    name: str
    phases: int = 3
    windings: int = 2
    buses: List[str] = field(default_factory=list)
    kvs: List[float] = field(default_factory=list)
    kvas: List[float] = field(default_factory=list)
    conns: List[str] = field(default_factory=list)
    taps: List[float] = field(default_factory=list)
    xhl: float = 0.0
    enabled: bool = True


@dataclass
class DSSCapacitor:
    """OpenDSS Capacitor definition."""
    name: str
    bus1: str
    phases: int = 3
    kvar: float = 0.0
    kv: float = 0.0
    enabled: bool = True


@dataclass
class DSSCircuit:
    """Complete OpenDSS Circuit definition."""
    name: str
    base_kv: float = 12.47
    buses: Dict[str, DSSBus] = field(default_factory=dict)
    lines: Dict[str, DSSLine] = field(default_factory=dict)
    loads: Dict[str, DSSLoad] = field(default_factory=dict)
    generators: Dict[str, DSSGenerator] = field(default_factory=dict)
    transformers: Dict[str, DSSTransformer] = field(default_factory=dict)
    capacitors: Dict[str, DSSCapacitor] = field(default_factory=dict)
    source_bus: str = "sourcebus"

    @property
    def n_buses(self) -> int:
        return len(self.buses)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def n_loads(self) -> int:
        return len(self.loads)

    @property
    def n_generators(self) -> int:
        return len(self.generators)

    @property
    def total_load_kw(self) -> float:
        return sum(load.kw for load in self.loads.values() if load.enabled)

    @property
    def total_gen_kw(self) -> float:
        return sum(gen.kw for gen in self.generators.values() if gen.enabled)

    def get_summary(self) -> Dict[str, Any]:
        """Get circuit summary."""
        return {
            "name": self.name,
            "base_kv": self.base_kv,
            "n_buses": self.n_buses,
            "n_lines": self.n_lines,
            "n_loads": self.n_loads,
            "n_generators": self.n_generators,
            "n_transformers": len(self.transformers),
            "n_capacitors": len(self.capacitors),
            "total_load_kw": self.total_load_kw,
            "total_gen_kw": self.total_gen_kw,
        }


class OpenDSSParser:
    """
    Parser for OpenDSS files.

    Supports the common DSS commands used in SMART-DS models:
    - Circuit definition
    - Lines, Loads, Generators
    - Transformers, Capacitors
    - Redirect/Compile commands
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize parser.

        Args:
            verbose: Print parsing progress
        """
        self.verbose = verbose
        self._circuit: Optional[DSSCircuit] = None
        self._base_path: Optional[Path] = None
        self._linecodes: Dict[str, Dict] = {}

    def parse_file(self, filepath: str) -> DSSCircuit:
        """
        Parse an OpenDSS file.

        Args:
            filepath: Path to .dss file

        Returns:
            DSSCircuit with parsed data
        """
        path = Path(filepath)
        self._base_path = path.parent
        self._circuit = DSSCircuit(name=path.stem)

        self._parse_dss_file(path)

        # Extract buses from line connections
        self._extract_buses()

        return self._circuit

    def _parse_dss_file(self, filepath: Path) -> None:
        """Parse a single DSS file."""
        if not filepath.exists():
            if self.verbose:
                print(f"Warning: File not found: {filepath}")
            return

        if self.verbose:
            print(f"Parsing: {filepath}")

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Remove comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # Split into commands (handle multi-line with ~)
        commands = self._split_commands(content)

        for cmd in commands:
            self._process_command(cmd.strip())

    def _split_commands(self, content: str) -> List[str]:
        """Split content into individual commands."""
        # Handle line continuations with ~
        content = re.sub(r'~\s*\n', ' ', content)

        # Split on newlines
        lines = content.split('\n')

        commands = []
        current_cmd = []

        for line in lines:
            line = line.strip()
            if not line:
                if current_cmd:
                    commands.append(' '.join(current_cmd))
                    current_cmd = []
            elif line.startswith('~'):
                # Continuation
                current_cmd.append(line[1:].strip())
            else:
                if current_cmd:
                    commands.append(' '.join(current_cmd))
                current_cmd = [line]

        if current_cmd:
            commands.append(' '.join(current_cmd))

        return commands

    def _process_command(self, cmd: str) -> None:
        """Process a single DSS command."""
        if not cmd:
            return

        cmd_lower = cmd.lower()

        # Handle different command types
        if cmd_lower.startswith('new circuit'):
            self._parse_circuit(cmd)
        elif cmd_lower.startswith('new line'):
            self._parse_line(cmd)
        elif cmd_lower.startswith('new load'):
            self._parse_load(cmd)
        elif cmd_lower.startswith('new generator') or cmd_lower.startswith('new pvsystem'):
            self._parse_generator(cmd)
        elif cmd_lower.startswith('new transformer'):
            self._parse_transformer(cmd)
        elif cmd_lower.startswith('new capacitor'):
            self._parse_capacitor(cmd)
        elif cmd_lower.startswith('new linecode'):
            self._parse_linecode(cmd)
        elif cmd_lower.startswith('redirect') or cmd_lower.startswith('compile'):
            self._handle_redirect(cmd)
        elif cmd_lower.startswith('set datapath'):
            self._handle_datapath(cmd)

    def _parse_params(self, cmd: str) -> Dict[str, str]:
        """Parse key=value parameters from command."""
        params = {}

        # Find all key=value pairs
        # Handle quoted values
        pattern = r'(\w+)\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|\[([^\]]+)\]|(\S+))'
        matches = re.findall(pattern, cmd, re.IGNORECASE)

        for match in matches:
            key = match[0].lower()
            value = match[1] or match[2] or match[3] or match[4]
            params[key] = value

        return params

    def _parse_circuit(self, cmd: str) -> None:
        """Parse circuit definition."""
        # New Circuit.name basekv=...
        match = re.search(r'circuit\.(\w+)', cmd, re.IGNORECASE)
        if match:
            self._circuit.name = match.group(1)

        params = self._parse_params(cmd)
        if 'basekv' in params:
            self._circuit.base_kv = float(params['basekv'])
        if 'bus1' in params:
            self._circuit.source_bus = params['bus1'].split('.')[0]

    def _parse_line(self, cmd: str) -> None:
        """Parse line definition."""
        # New Line.name bus1=... bus2=...
        match = re.search(r'line\.(\w+)', cmd, re.IGNORECASE)
        if not match:
            return

        name = match.group(1)
        params = self._parse_params(cmd)

        line = DSSLine(
            name=name,
            bus1=params.get('bus1', '').split('.')[0],
            bus2=params.get('bus2', '').split('.')[0],
            length=float(params.get('length', 1.0)),
            units=params.get('units', 'km'),
            r1=float(params.get('r1', 0)),
            x1=float(params.get('x1', 0)),
            normamps=float(params.get('normamps', 400)),
            phases=int(params.get('phases', 3)),
            linecode=params.get('linecode', ''),
        )

        # Apply linecode if specified
        if line.linecode and line.linecode in self._linecodes:
            lc = self._linecodes[line.linecode]
            if 'r1' in lc and line.r1 == 0:
                line.r1 = lc['r1']
            if 'x1' in lc and line.x1 == 0:
                line.x1 = lc['x1']
            if 'normamps' in lc:
                line.normamps = lc['normamps']

        self._circuit.lines[name] = line

    def _parse_load(self, cmd: str) -> None:
        """Parse load definition."""
        match = re.search(r'load\.(\w+)', cmd, re.IGNORECASE)
        if not match:
            return

        name = match.group(1)
        params = self._parse_params(cmd)

        load = DSSLoad(
            name=name,
            bus1=params.get('bus1', '').split('.')[0],
            phases=int(params.get('phases', 3)),
            kv=float(params.get('kv', 0)),
            kw=float(params.get('kw', 0)),
            kvar=float(params.get('kvar', 0)),
            pf=float(params.get('pf', 0.95)),
            model=int(params.get('model', 1)),
            conn=params.get('conn', 'wye'),
        )

        self._circuit.loads[name] = load

    def _parse_generator(self, cmd: str) -> None:
        """Parse generator or PV system definition."""
        match = re.search(r'(generator|pvsystem)\.(\w+)', cmd, re.IGNORECASE)
        if not match:
            return

        gen_type = match.group(1).lower()
        name = match.group(2)
        params = self._parse_params(cmd)

        gen = DSSGenerator(
            name=name,
            bus1=params.get('bus1', '').split('.')[0],
            phases=int(params.get('phases', 3)),
            kv=float(params.get('kv', 0)),
            kw=float(params.get('kw', params.get('pmpp', 0))),
            kvar=float(params.get('kvar', 0)),
            pf=float(params.get('pf', 1.0)),
            gen_type='pv' if gen_type == 'pvsystem' else 'gen',
        )

        self._circuit.generators[name] = gen

    def _parse_transformer(self, cmd: str) -> None:
        """Parse transformer definition."""
        match = re.search(r'transformer\.(\w+)', cmd, re.IGNORECASE)
        if not match:
            return

        name = match.group(1)
        params = self._parse_params(cmd)

        # Parse bus list
        buses = []
        if 'buses' in params:
            buses = [b.strip().split('.')[0] for b in params['buses'].split(',')]

        # Parse kV list
        kvs = []
        if 'kvs' in params:
            kvs = [float(k.strip()) for k in params['kvs'].split(',')]

        # Parse kVA list
        kvas = []
        if 'kvas' in params:
            kvas = [float(k.strip()) for k in params['kvas'].split(',')]

        xfmr = DSSTransformer(
            name=name,
            phases=int(params.get('phases', 3)),
            windings=int(params.get('windings', 2)),
            buses=buses,
            kvs=kvs,
            kvas=kvas,
            xhl=float(params.get('xhl', params.get('%r', 0))),
        )

        self._circuit.transformers[name] = xfmr

    def _parse_capacitor(self, cmd: str) -> None:
        """Parse capacitor definition."""
        match = re.search(r'capacitor\.(\w+)', cmd, re.IGNORECASE)
        if not match:
            return

        name = match.group(1)
        params = self._parse_params(cmd)

        cap = DSSCapacitor(
            name=name,
            bus1=params.get('bus1', '').split('.')[0],
            phases=int(params.get('phases', 3)),
            kvar=float(params.get('kvar', 0)),
            kv=float(params.get('kv', 0)),
        )

        self._circuit.capacitors[name] = cap

    def _parse_linecode(self, cmd: str) -> None:
        """Parse linecode definition."""
        match = re.search(r'linecode\.(\w+)', cmd, re.IGNORECASE)
        if not match:
            return

        name = match.group(1)
        params = self._parse_params(cmd)

        self._linecodes[name] = {
            'r1': float(params.get('r1', 0)),
            'x1': float(params.get('x1', 0)),
            'normamps': float(params.get('normamps', 400)),
        }

    def _handle_redirect(self, cmd: str) -> None:
        """Handle redirect/compile command."""
        match = re.search(r'(?:redirect|compile)\s+["\']?([^"\']+)["\']?', cmd, re.IGNORECASE)
        if match:
            filename = match.group(1).strip()
            filepath = self._base_path / filename
            if filepath.exists():
                self._parse_dss_file(filepath)
            elif self.verbose:
                print(f"Warning: Redirect file not found: {filepath}")

    def _handle_datapath(self, cmd: str) -> None:
        """Handle set datapath command."""
        match = re.search(r'datapath\s*=\s*["\']?([^"\']+)["\']?', cmd, re.IGNORECASE)
        if match:
            datapath = match.group(1).strip()
            new_path = self._base_path / datapath
            if new_path.exists():
                self._base_path = new_path

    def _extract_buses(self) -> None:
        """Extract bus definitions from line connections."""
        buses = set()

        # From lines
        for line in self._circuit.lines.values():
            if line.bus1:
                buses.add(line.bus1)
            if line.bus2:
                buses.add(line.bus2)

        # From loads
        for load in self._circuit.loads.values():
            if load.bus1:
                buses.add(load.bus1)

        # From generators
        for gen in self._circuit.generators.values():
            if gen.bus1:
                buses.add(gen.bus1)

        # From transformers
        for xfmr in self._circuit.transformers.values():
            for bus in xfmr.buses:
                if bus:
                    buses.add(bus)

        # Add source bus
        buses.add(self._circuit.source_bus)

        # Create bus objects
        for bus_name in buses:
            if bus_name and bus_name not in self._circuit.buses:
                self._circuit.buses[bus_name] = DSSBus(
                    name=bus_name,
                    base_kv=self._circuit.base_kv
                )


def parse_smartds(
    directory: str,
    master_file: str = "Master.dss",
    verbose: bool = False
) -> DSSCircuit:
    """
    Parse a SMART-DS directory.

    Args:
        directory: Path to SMART-DS data directory
        master_file: Name of master DSS file
        verbose: Print parsing progress

    Returns:
        Parsed DSSCircuit
    """
    parser = OpenDSSParser(verbose=verbose)
    master_path = Path(directory) / master_file

    if not master_path.exists():
        # Try to find any .dss file
        dss_files = list(Path(directory).glob("*.dss"))
        if dss_files:
            master_path = dss_files[0]
        else:
            raise FileNotFoundError(f"No DSS files found in {directory}")

    return parser.parse_file(str(master_path))
