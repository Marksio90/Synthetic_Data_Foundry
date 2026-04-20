"""
agents/crawlers — Pluggable Source Crawler Package for Gap Scout (ENTERPRISE EDITION)

Ten pakiet stanowi zaawansowaną warstwę akwizycji danych (Data Ingestion Layer).
Implementuje architekturę wielowarstwową (Layer A-E) do inteligentnego pobierania, 
analizy i deduplikacji treści ze źródeł zewnętrznych.

Architektura Warstw:
  - Layer A: Oficjalne rejestry i dzienniki prawne (np. EUR-Lex).
  - Layer B: Instytucje nadzorcze i regulacyjne (np. ESMA, EBA, KNF).
  - Layer C: Renomowane organizacje doradcze i audytorskie (Wielka Czwórka).
  - Layer D: Portale branżowe i wiadomości finansowe (News Feeds).
  - Layer E: Alternatywne i niszowe źródła wiedzy ESG.

Komponenty Główne:
  - WebSubSubscriber: Nasłuchiwanie na zdarzenia w czasie rzeczywistym (Push).
  - PollingScheduler: Cykliczne odpytywanie źródeł asynchronicznych (Pull).
  - DedupPipeline: Moduł eliminujący powielone informacje przed wpuszczeniem do bazy.
"""

from __future__ import annotations

import logging

# ---------------------------------------------------------------------------
# Importy Komponentów Bazowych i Infrastruktury
# ---------------------------------------------------------------------------
from agents.crawlers.base import CrawlerBase
from agents.crawlers.dedup import DedupPipeline
from agents.crawlers.scheduler import PollingScheduler
from agents.crawlers.websub import WebSubSubscriber

# ---------------------------------------------------------------------------
# Importy Warstw Pobierania (Data Layers A-E)
# ---------------------------------------------------------------------------
from agents.crawlers.layer_a import get_crawler_status, run_layer_a
from agents.crawlers.layer_b import get_crawler_status_b, run_layer_b
from agents.crawlers.layer_c import get_crawler_status_c, run_layer_c
from agents.crawlers.layer_d import get_crawler_status_d, run_layer_d
from agents.crawlers.layer_e import get_crawler_status_e, run_layer_e

# Inicjalizacja bazowego loggera dla całego pakietu crawlerów.
# Submoduły powinny używać: logger = logging.getLogger(__name__)
logger = logging.getLogger("foundry.agents.crawlers")
logger.debug("Inicjalizacja modułu akwizycji danych (Gap Scout Crawlers)...")


# ---------------------------------------------------------------------------
# Eksporty Pakietu (Fasada API)
# Zdefiniowanie __all__ chroni przed importowaniem wewnętrznych zmiennych 
# przy użyciu 'from agents.crawlers import *'
# ---------------------------------------------------------------------------
__all__ = [
    # Infrastruktura CORE
    "CrawlerBase",
    "DedupPipeline",
    "PollingScheduler",
    "WebSubSubscriber",
    
    # Interfejsy Layer A (Rejestry Prawne)
    "run_layer_a", 
    "get_crawler_status",
    
    # Interfejsy Layer B (Regulatorzy)
    "run_layer_b", 
    "get_crawler_status_b",
    
    # Interfejsy Layer C (Doradztwo/Audyt)
    "run_layer_c", 
    "get_crawler_status_c",
    
    # Interfejsy Layer D (Wiadomości)
    "run_layer_d", 
    "get_crawler_status_d",
    
    # Interfejsy Layer E (Źródła Alternatywne)
    "run_layer_e", 
    "get_crawler_status_e",
]
