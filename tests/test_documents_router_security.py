import os
import unittest

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/db")
os.environ.setdefault("ASYNC_DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/db")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from fastapi import FastAPI
from fastapi.routing import APIRoute

from api.routers import documents
from api.security import require_admin_api_key


class DocumentsRouterSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        app = FastAPI()
        app.include_router(documents.router, prefix="/api/documents")
        self._routes: list[APIRoute] = [
            r for r in app.routes if isinstance(r, APIRoute)
        ]

    def _route(self, path: str, method: str) -> APIRoute:
        for route in self._routes:
            if route.path == path and method in route.methods:
                return route
        self.fail(f"Route {method} {path} not found")

    def _has_admin_dependency(self, route: APIRoute) -> bool:
        return any(
            dep.call is require_admin_api_key
            for dep in route.dependant.dependencies
        )

    def test_get_documents_is_public(self) -> None:
        route = self._route("/api/documents", "GET")
        self.assertFalse(self._has_admin_dependency(route))

    def test_mutating_document_routes_require_admin_key(self) -> None:
        upload_route = self._route("/api/documents/upload", "POST")
        delete_route = self._route("/api/documents/{filename}", "DELETE")

        self.assertTrue(self._has_admin_dependency(upload_route))
        self.assertTrue(self._has_admin_dependency(delete_route))


if __name__ == "__main__":
    unittest.main()
