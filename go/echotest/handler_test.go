package learn_echotest

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/labstack/echo/v4"
	"github.com/stretchr/testify/assert"
)

var (
	mockDB = map[string]*User{
		"jon@labstack.com": {"Jon Snow", "jon@labstack.com"},
	}
	userJSON = "{\"name\":\"Jon Snow\",\"email\":\"jon@labstack.com\"}\n"
)

func TestCreateUser(t *testing.T) {
	// Setup
	e := echo.New()
	fmt.Println(strings.NewReader(userJSON))
	req := httptest.NewRequest(http.MethodPost, "/", strings.NewReader(userJSON))
	req.Header.Set(echo.HeaderContentType, echo.MIMEApplicationJSON)
	rec := httptest.NewRecorder()
	c := e.NewContext(req, rec)
	h := &handler{mockDB}

	// Assertions
	if assert.NoError(t, h.createUser(c)) {
		assert.Equal(t, http.StatusCreated, rec.Code)
		assert.Equal(t, userJSON, rec.Body.String())
	}
}

func TestGetUser(t *testing.T) {
	// Setup
	e := echo.New()
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	c := e.NewContext(req, rec)
	c.SetPath("/users/:email")
	c.SetParamNames("email")
	c.SetParamValues("jon@labstack.com")
	h := &handler{mockDB}

	// Assertions
	if assert.NoError(t, h.getUser(c)) {
		assert.Equal(t, http.StatusOK, rec.Code)
		assert.Equal(t, userJSON, rec.Body.String())
	}
}
