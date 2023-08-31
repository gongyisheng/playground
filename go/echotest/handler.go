package learn_echotest

import (
	_ "fmt"
	"net/http"

	"github.com/labstack/echo/v4"
)

type User struct {
	Name  string `json:"name" form:"name"`
	Email string `json:"email" form:"email"`
}
type handler struct {
	db map[string]*User
}

func (h *handler) createUser(c echo.Context) error {
	u := new(User)
	if err := c.Bind(u); err != nil {
		return err
	}
	return c.JSON(http.StatusCreated, u)
}

func (h *handler) getUser(c echo.Context) error {
	email := c.Param("email")
	user := h.db[email]
	if user == nil {
		return echo.NewHTTPError(http.StatusNotFound, "user not found")
	}
	return c.JSON(http.StatusOK, user)
}
