from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
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


@dataclass
class InMemoryMetadata:
    distance: Optional[float] = None


@dataclass
class InMemoryQueryObject:
    uuid: str
    properties: Dict[str, Any]
    metadata: InMemoryMetadata


class InMemoryQueryObjectsResult:
    def __init__(self, objects: List[InMemoryQueryObject]):
        self.objects = objects


class InMemoryCollectionData:
    def __init__(self, storage: Dict[str, Dict[str, Any]], collection_name: str):
        self.storage = storage
        self.collection_name = collection_name

    def insert(self, properties: Dict[str, Any], vector: List[float]) -> str:
        object_id = str(uuid.uuid4())
        self.storage[object_id] = {
            "collection_name": self.collection_name,
            "data_object": dict(properties),
            "vector": list(vector),
        }
        return object_id

    def delete_by_id(self, uuid: uuid.UUID) -> None:
        self.storage.pop(str(uuid), None)

    def update(
        self,
        uuid: uuid.UUID,
        properties: Dict[str, Any],
        vector: List[float],
    ) -> None:
        object_id = str(uuid)
        if object_id not in self.storage:
            raise KeyError(object_id)
        self.storage[object_id]["data_object"] = dict(properties)
        self.storage[object_id]["vector"] = list(vector)


class InMemoryCollectionQuery:
    def __init__(self, storage: Dict[str, Dict[str, Any]], collection_name: str):
        self.storage = storage
        self.collection_name = collection_name

    def _extract_filter(self, filters: Any) -> tuple[Optional[List[str]], Any]:
        if filters is None:
            return None, None
        if isinstance(filters, dict):
            return filters.get("path"), filters.get("valueInt", filters.get("valueText", filters.get("value")))

        path = getattr(filters, "path", None)
        value = None
        for attr in ("valueInt", "value_int", "valueText", "value_text", "valueBoolean", "valueBoolean", "value"):
            if hasattr(filters, attr):
                value = getattr(filters, attr)
                if value is not None:
                    break
        if value is None:
            value = getattr(filters, "value", None)
        return path, value

    def _matches(self, payload: Dict[str, Any], filters: Any) -> bool:
        path, value = self._extract_filter(filters)
        if not path:
            return True
        if path == ["sqlite_id"]:
            return payload["data_object"].get("sqlite_id") == value
        if path == ["vector_layer"]:
            return payload["data_object"].get("vector_layer") == value
        return True

    def fetch_objects(self, filters: Any = None, limit: Optional[int] = None, return_metadata: Any = None) -> InMemoryQueryObjectsResult:
        objects: List[InMemoryQueryObject] = []
        for object_id, payload in self.storage.items():
            if payload["collection_name"] != self.collection_name:
                continue
            if not self._matches(payload, filters):
                continue
            objects.append(
                InMemoryQueryObject(
                    uuid=object_id,
                    properties=dict(payload["data_object"]),
                    metadata=InMemoryMetadata(distance=None),
                )
            )
            if limit is not None and len(objects) >= limit:
                break
        return InMemoryQueryObjectsResult(objects)

    def fetch_object_by_id(self, uuid: uuid.UUID) -> Optional[InMemoryQueryObject]:
        object_id = str(uuid)
        payload = self.storage.get(object_id)
        if not payload or payload["collection_name"] != self.collection_name:
            return None
        return InMemoryQueryObject(
            uuid=object_id,
            properties=dict(payload["data_object"]),
            metadata=InMemoryMetadata(distance=None),
        )

    def near_vector(self, near_vector: List[float], limit: int, filters: Any = None, return_metadata: Any = None) -> InMemoryQueryObjectsResult:
        objects = self.fetch_objects(filters=filters, limit=None, return_metadata=return_metadata).objects
        scored: List[tuple[float, InMemoryQueryObject]] = []
        for obj in objects:
            vector = self.storage[obj.uuid]["vector"]
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vector, near_vector)))
            scored.append((distance, InMemoryQueryObject(uuid=obj.uuid, properties=obj.properties, metadata=InMemoryMetadata(distance=distance))))
        scored.sort(key=lambda item: item[0])
        return InMemoryQueryObjectsResult([item[1] for item in scored[:limit]])


class InMemoryCollection:
    def __init__(self, storage: Dict[str, Dict[str, Any]], name: str):
        self.name = name
        self.data = InMemoryCollectionData(storage, name)
        self.query = InMemoryCollectionQuery(storage, name)


class InMemoryCollectionsManager:
    def __init__(self, storage: Dict[str, Dict[str, Any]]):
        self.storage = storage
        self.collections: Dict[str, InMemoryCollection] = {}

    def exists(self, name: str) -> bool:
        return name in self.collections

    def create(self, name: str, description: str = "", properties: Optional[List[Dict[str, Any]]] = None) -> None:
        if name not in self.collections:
            self.collections[name] = InMemoryCollection(self.storage, name)

    def get(self, name: str) -> InMemoryCollection:
        if name not in self.collections:
            raise KeyError(name)
        return self.collections[name]


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
        self.collections = InMemoryCollectionsManager(self.storage)
        self.data_object = InMemoryDataObject(self.storage)
        self.query = InMemoryQuery(self.storage)


