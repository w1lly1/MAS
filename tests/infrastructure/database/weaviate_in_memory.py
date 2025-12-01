from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional


class InMemorySchema:
    def __init__(self):
        self._classes: List[Dict[str, Any]] = []

    def get(self) -> Dict[str, Any]:
        return {"classes": self._classes}

    def create_class(self, class_obj: Dict[str, Any]) -> None:
        class_names = {c["class"] for c in self._classes}
        if class_obj["class"] not in class_names:
            self._classes.append(class_obj)


class InMemoryDataObject:
    def __init__(self, storage: Dict[str, Dict[str, Any]]):
        self.storage = storage

    def create(self, data_object: Dict[str, Any], class_name: str, vector: List[float]) -> str:
        object_id = str(uuid.uuid4())
        self.storage[object_id] = {
            "class_name": class_name,
            "data_object": data_object,
            "vector": vector,
        }
        return object_id

    def update(
        self,
        data_object: Dict[str, Any],
        class_name: str,
        uuid: str,
        vector: List[float],
    ) -> None:
        if uuid not in self.storage:
            raise KeyError(uuid)
        self.storage[uuid]["data_object"] = data_object
        self.storage[uuid]["vector"] = vector

    def delete(self, uuid: str, class_name: str) -> None:
        self.storage.pop(uuid, None)


class InMemoryQueryGet:
    def __init__(self, storage: Dict[str, Dict[str, Any]], class_name: str):
        self.storage = storage
        self.class_name = class_name
        self.where_filter: Optional[Dict[str, Any]] = None
        self.limit: Optional[int] = None
        self.fields: List[str] = []

    def with_where(self, where_filter: Dict[str, Any]) -> "InMemoryQueryGet":
        self.where_filter = where_filter
        return self

    def with_limit(self, limit: int) -> "InMemoryQueryGet":
        self.limit = limit
        return self

    def do(self) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        include_additional = any("_additional" in field for field in self.fields)

        for object_id, payload in self.storage.items():
            if payload["class_name"] != self.class_name:
                continue
            if self.where_filter:
                path = self.where_filter.get("path", [])
                if path == ["sqlite_id"]:
                    value = payload["data_object"].get("sqlite_id")
                    if value != self.where_filter.get("valueInt"):
                        continue
            entry: Dict[str, Any] = {}
            for field in self.fields:
                if field.startswith("_additional"):
                    continue
                entry[field] = payload["data_object"].get(field)
            if include_additional:
                entry["_additional"] = {"id": object_id}
            results.append(entry)
            if self.limit is not None and len(results) >= self.limit:
                break

        return {"data": {"Get": {self.class_name: results}}}


class InMemoryQuery:
    def __init__(self, storage: Dict[str, Dict[str, Any]]):
        self.storage = storage

    def get(self, class_name: str, fields: List[str]) -> InMemoryQueryGet:
        query = InMemoryQueryGet(storage=self.storage, class_name=class_name)
        query.fields = fields
        return query


class InMemoryWeaviateClient:
    def __init__(self):
        self.storage: Dict[str, Dict[str, Any]] = {}
        self.schema = InMemorySchema()
        self.data_object = InMemoryDataObject(self.storage)
        self.query = InMemoryQuery(self.storage)


