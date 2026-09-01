from supabase import AsyncClient, Client, acreate_client, create_client

from optexity.utils.settings import settings

__all__ = ["Client", "AsyncClient", "get_client", "get_async_client"]

_client = None
_admin_client = None
_async_client = None
_async_admin_client = None


def get_client(admin: bool = False) -> Client:
    """
    Get or create a Supabase client instance
    """
    global _client
    global _admin_client

    if admin:
        if _admin_client is None:
            _admin_client = create_client(
                settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY
            )
        return _admin_client
    else:
        if _client is None:
            _client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        return _client


async def get_async_client(admin: bool = False) -> AsyncClient:
    """
    Get or create an async Supabase client instance
    """
    global _async_client
    global _async_admin_client

    if admin:
        if _async_admin_client is None:
            _async_admin_client = await acreate_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_SERVICE_ROLE_KEY,
            )
        return _async_admin_client
    else:
        if _async_client is None:
            _async_client = await acreate_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_KEY,
            )
        return _async_client
