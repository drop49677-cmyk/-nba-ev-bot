import supabase._sync.client
from supabase import create_client

original_init = supabase._sync.client.SyncClient.__init__

def patched_init(self, supabase_url, supabase_key, options=None):
    if options is None:
        options = supabase._sync.client.ClientOptions()
    if supabase_key.startswith("sb_"):
        original_init(self, supabase_url, "eyJhbGci.eyJzdWIi.fake", options)
        self.supabase_key = supabase_key
        self._auth_token = {"Authorization": f"Bearer {supabase_key}"}
        self.options.headers.update(self._auth_token)
    else:
        original_init(self, supabase_url, supabase_key, options)

supabase._sync.client.SyncClient.__init__ = patched_init

c = create_client("https://qbwgqtmdrmlkscvnuiif.supabase.co", "sb_publishable_fU2lvnwLrmptNTqNgorcZQ_UbJnwQxP")
print("SUCCESS:", c.supabase_key)
